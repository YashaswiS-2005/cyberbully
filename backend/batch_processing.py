"""
Batch Processing Module
Handles large-scale detection with async processing and batch operations.
"""

import os
import logging
import time
import threading
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from queue import Queue
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process large datasets in batches for memory efficiency."""

    def __init__(self, model_pipeline, batch_size: int = 1000):
        self.model_pipeline = model_pipeline
        self.batch_size = batch_size
        self.results = []
        self.errors = []

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Process a DataFrame in batches."""
        total_rows = len(df)
        predictions = []
        confidences = []
        
        for start_idx in range(0, total_rows, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_rows)
            batch = df.iloc[start_idx:end_idx][text_column].fillna("").astype(str)
            
            try:
                batch_preds = self.model_pipeline.predict(batch.tolist())
                batch_probs = self.model_pipeline.predict_proba(batch.tolist())
                batch_conf = [float(max(probs)) * 100 for probs in batch_probs]
                
                predictions.extend(batch_preds)
                confidences.extend(batch_conf)
                
                if progress_callback:
                    progress_callback(end_idx, total_rows)
                    
            except Exception as e:
                logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
                # Fill with error values
                predictions.extend(["error"] * len(batch))
                confidences.extend([0.0] * len(batch))
                self.errors.append({"batch": f"{start_idx}-{end_idx}", "error": str(e)})

        df["prediction"] = predictions
        df["confidence"] = confidences
        df["processed_at"] = datetime.utcnow().isoformat()
        
        return df

    def process_csv(
        self,
        input_path: str,
        output_path: str,
        text_column: str = "text",
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Process a CSV file in batches and save results."""
        logger.info(f"Processing CSV: {input_path}")
        
        # Read in chunks to handle large files
        chunk_results = []
        total_processed = 0
        
        for chunk in pd.read_csv(input_path, chunksize=self.batch_size):
            if text_column not in chunk.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV")
            
            processed_chunk = self.process_dataframe(
                chunk,
                text_column=text_column,
                progress_callback=progress_callback
            )
            chunk_results.append(processed_chunk)
            total_processed += len(chunk)
            logger.info(f"Processed {total_processed} rows")

        # Combine all chunks
        result_df = pd.concat(chunk_results, ignore_index=True)
        
        # Save results
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved results to: {output_path}")
        
        return output_path


class AsyncDetectionQueue:
    """Queue-based async detection for real-time processing."""

    def __init__(self, model_pipeline, max_workers: int = 4):
        self.model_pipeline = model_pipeline
        self.input_queue = Queue(maxsize=10000)
        self.output_queue = Queue()
        self.max_workers = max_workers
        self.workers = []
        self.running = False

    def start(self):
        """Start the worker threads."""
        if self.running:
            return
        
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.max_workers} async workers")

    def stop(self):
        """Stop the worker threads."""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)
        self.workers.clear()
        logger.stopped("async workers")

    def _worker(self):
        """Worker thread function."""
        while self.running:
            try:
                item = self.input_queue.get(timeout=1)
                if item is None:  # Shutdown signal
                    break

                request_id, text = item
                try:
                    prediction = self.model_pipeline.predict([text])[0]
                    probabilities = self.model_pipeline.predict_proba([text])[0]
                    confidence = float(max(probabilities)) * 100
                    
                    result = {
                        "request_id": request_id,
                        "text": text,
                        "prediction": prediction,
                        "confidence": confidence,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    result = {
                        "request_id": request_id,
                        "text": text,
                        "prediction": "error",
                        "confidence": 0.0,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                self.output_queue.put(result)
                self.input_queue.task_done()
                
            except Exception:
                continue

    def submit(self, request_id: str, text: str):
        """Submit a text for async detection."""
        self.input_queue.put((request_id, text))

    def get_result(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get a result from the output queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Exception:
            return None

    def get_pending_count(self) -> int:
        """Get the number of pending items."""
        return self.input_queue.qsize()


class StreamingDetector:
    """Process streaming data from social media collectors."""

    def __init__(self, model_pipeline, batch_processor: BatchProcessor):
        self.model_pipeline = model_pipeline
        self.batch_processor = batch_processor
        self.stats = {
            "total_processed": 0,
            "by_label": {},
            "avg_confidence": 0.0
        }

    def process_stream(
        self,
        data_generator,
        output_path: Optional[str] = None,
        flush_interval: int = 1000
    ) -> pd.DataFrame:
        """Process a stream of data."""
        buffer = []
        all_results = []

        for item in data_generator:
            # Extract text from various formats
            text = item.get("text") or item.get("title") or item.get("body") or ""
            
            if text:
                buffer.append(text)
                
                if len(buffer) >= flush_interval:
                    # Process batch
                    batch_df = pd.DataFrame({ "text": buffer })
                    results = self.batch_processor.process_dataframe(batch_df)
                    all_results.append(results)
                    buffer = []

        # Process remaining items
        if buffer:
            batch_df = pd.DataFrame({ "text": buffer })
            results = self.batch_processor.process_dataframe(batch_df)
            all_results.append(results)

        # Combine all results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            self._update_stats(final_df)
            
            if output_path:
                final_df.to_csv(output_path, index=False)
            
            return final_df
        
        return pd.DataFrame()

    def _update_stats(self, df: pd.DataFrame):
        """Update processing statistics."""
        self.stats["total_processed"] = len(df)
        
        for label in df["prediction"].unique():
            count = len(df[df["prediction"] == label])
            self.stats["by_label"][label] = count
        
        if "confidence" in df.columns:
            self.stats["avg_confidence"] = df["confidence"].mean()

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats


class LargeDatasetHandler:
    """Handle very large datasets with sampling and distributed processing."""

    def __init__(self, model_pipeline):
        self.model_pipeline = model_pipeline
        self.sample_size = 10000

    def sample_dataset(
        self,
        input_path: str,
        sample_size: Optional[int] = None,
        strategy: str = "random"
    ) -> pd.DataFrame:
        """Sample a large dataset for initial analysis."""
        size = sample_size or self.sample_size
        
        # First, get total row count without loading full file
        total_rows = sum(1 for _ in open(input_path, encoding="utf-8")) - 1  # subtract header
        
        if total_rows <= size:
            return pd.read_csv(input_path)

        if strategy == "random":
            # Random sampling
            skiprows = sorted(set(range(2, total_rows + 1)) - 
                            set(np.random.choice(range(2, total_rows + 1), 
                                                size=total_rows - size, 
                                                replace=False)))
            return pd.read_csv(input_path, skiprows=skiprows)
        
        elif strategy == "first":
            # Take first N rows
            return pd.read_csv(input_path, nrows=size)
        
        elif strategy == "stratified":
            # Stratified sampling based on label distribution
            df = pd.read_csv(input_path)
            if "label" in df.columns:
                return df.groupby("label", group_keys=False).apply(
                    lambda x: x.sample(frac=size/len(df), random_state=42)
                )
            return df.sample(n=size, random_state=42)

    def analyze_distribution(self, input_path: str) -> Dict[str, Any]:
        """Analyze the distribution of a large dataset."""
        stats = {
            "total_rows": 0,
            "label_distribution": {},
            "text_length_stats": {}
        }
        
        # Count rows and analyze in chunks
        label_counts = {}
        text_lengths = []
        
        for chunk in pd.read_csv(input_path, chunksize=10000):
            stats["total_rows"] += len(chunk)
            
            if "label" in chunk.columns:
                for label, count in chunk["label"].value_counts().items():
                    label_counts[label] = label_counts.get(label, 0) + count
            
            if "text" in chunk.columns:
                text_lengths.extend(chunk["text"].str.len().tolist())

        stats["label_distribution"] = label_counts
        if text_lengths:
            stats["text_length_stats"] = {
                "mean": np.mean(text_lengths),
                "median": np.median(text_lengths),
                "max": np.max(text_lengths),
                "min": np.min(text_lengths)
            }

        return stats