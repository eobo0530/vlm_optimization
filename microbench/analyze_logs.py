#!/usr/bin/env python3
"""
Log Analyzer for VLM Performance Benchmarks
Parses profiling logs to extract Vision/Prefill/Decode breakdown.

Usage:
    python analyze_logs.py log1.txt log2.txt ...
"""

import re
import numpy as np
import argparse
import os


def parse_log(file_path):
    """Parse a benchmark log file and extract profiling metrics."""
    print(f"Parsing {file_path}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    vision_times = []
    prefill_times = []
    decode_times = []
    first_decode_times = [] 
    vision_plus_prefill_times = []
    
    sample_vision = []
    sample_decodes = []
    
    # State machine to parse per-sample metrics
    for line in lines:
        if "[Profiling] VisionStep:" in line:
            # Flush previous sample's decodes if pending
            if sample_decodes:
                current_prefill = 0
                current_vision_sum = 0
                
                if len(sample_decodes) > 0:
                    current_prefill = sample_decodes[0]
                    prefill_times.append(current_prefill)
                    if len(sample_decodes) > 1:
                        first_decode_times.append(sample_decodes[1])
                        decode_times.extend(sample_decodes[1:])
                
                if sample_vision:
                    current_vision_sum = sum(sample_vision)
                    vision_times.append(current_vision_sum)
                
                if current_prefill > 0: 
                    vision_plus_prefill_times.append(current_vision_sum + current_prefill)
                
                sample_vision = []
                sample_decodes = []
            
            match = re.search(r"VisionStep:\s+([\d\.]+)\s+ms", line)
            if match:
                sample_vision.append(float(match.group(1)))
                
        elif "[Profiling] LLM_Decode:" in line:
            match = re.search(r"LLM_Decode:\s+([\d\.]+)\s+ms", line)
            if match:
                sample_decodes.append(float(match.group(1)))

    # Flush final sample
    if sample_decodes:
        current_prefill = 0
        current_vision_sum = 0
        
        if len(sample_decodes) > 0:
            current_prefill = sample_decodes[0]
            prefill_times.append(current_prefill)
            if len(sample_decodes) > 1:
                first_decode_times.append(sample_decodes[1])
                decode_times.extend(sample_decodes[1:])
        if sample_vision:
            current_vision_sum = sum(sample_vision)
            vision_times.append(current_vision_sum)
            
        if current_prefill > 0:
            vision_plus_prefill_times.append(current_vision_sum + current_prefill)

    return {
        "Vision (Avg)": np.mean(vision_times) if vision_times else 0,
        "Vision (P50)": np.percentile(vision_times, 50) if vision_times else 0,
        "LLM Prefill (Avg)": np.mean(prefill_times) if prefill_times else 0,
        "LLM Prefill (P50)": np.percentile(prefill_times, 50) if prefill_times else 0,
        "Vision+Prefill (Avg)": np.mean(vision_plus_prefill_times) if vision_plus_prefill_times else 0,
        "First Decode (Avg)": np.mean(first_decode_times) if first_decode_times else 0,
        "Avg Decode (Avg)": np.mean(decode_times) if decode_times else 0, 
        "Count": len(prefill_times)
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark logs for Vision/Prefill/Decode breakdown")
    parser.add_argument("log_files", nargs='+', help="Log files to analyze")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save analysis results (default: same as first log file)")
    args = parser.parse_args()

    full_report = ""

    for log_file in args.log_files:
        if not os.path.exists(log_file):
            print(f"File not found: {log_file}")
            continue
            
        stats = parse_log(log_file)
        
        # Build report string
        report_chunk = f"\n{'='*50}\n"
        report_chunk += f"Analysis for: {log_file}\n"
        report_chunk += f"Samples Processed: {stats['Count']}\n"
        report_chunk += f"{'-'*50}\n"
        
        if stats['Count'] > 0:
            report_chunk += f"Vision Latency:      {stats['Vision (Avg)']:.2f} ms (Avg) | {stats['Vision (P50)']:.2f} ms (P50)\n"
            report_chunk += f"LLM Prefill:         {stats['LLM Prefill (Avg)']:.2f} ms (Avg) | {stats['LLM Prefill (P50)']:.2f} ms (P50)\n"
            report_chunk += f"Vision + Prefill:    {stats['Vision+Prefill (Avg)']:.2f} ms (Avg)\n"
            report_chunk += f"First Decode Step:   {stats['First Decode (Avg)']:.2f} ms (Avg)\n"
            report_chunk += f"Avg Decode Step:     {stats['Avg Decode (Avg)']:.2f} ms (Avg) (All steps)\n"
        else:
            report_chunk += "No profiling data found in log file.\n"
            report_chunk += "Make sure ENABLE_PROFILING=1 is set and the log contains:\n"
            report_chunk += "  [Profiling] VisionStep: X.XX ms\n"
            report_chunk += "  [Profiling] LLM_Decode: X.XX ms\n"
        
        report_chunk += f"{'='*50}\n"
        
        print(report_chunk, end="")
        full_report += report_chunk

    # Save to file
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(args.log_files[0])) if args.log_files else "."
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "analysis_results.txt")
    with open(output_file, "w") as f:
        f.write(full_report)
    print(f"\n[Info] Results saved to {output_file}")


if __name__ == "__main__":
    main()
