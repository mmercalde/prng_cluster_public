#!/usr/bin/env python3
"""
Lottery Data Analysis with Multiple Conversion Methods
"""
import json
import sys
import hashlib

def convert_lottery_to_32bit(draws, method='hash'):
    """Convert lottery draws using different methods"""
    converted = []
    
    if method == 'multiply':
        # Original method - simple scaling
        for draw in draws:
            if isinstance(draw, str):
                draw = int(draw)
            converted_val = (draw * 4294967) & 0xFFFFFFFF
            converted.append(converted_val)
            
    elif method == 'hash':
        # Hash-based conversion - best entropy preservation
        for draw in draws:
            draw_str = str(draw)
            hash_obj = hashlib.md5(draw_str.encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)  # Take first 8 hex chars
            converted.append(hash_int)
            
    elif method == 'shift_combine':
        # Bit-shift combination method
        for i, draw in enumerate(draws):
            if isinstance(draw, str):
                draw = int(draw)
            # Combine draw number with its position
            combined = ((draw & 0xFFF) << 20) | ((i & 0xFFF) << 8) | (draw & 0xFF)
            converted.append(combined & 0xFFFFFFFF)
            
    elif method == 'sequence':
        # Sequential transformation preserving order
        for i, draw in enumerate(draws):
            if isinstance(draw, str):
                draw = int(draw)
            # Use position and value together
            seq_val = (i << 16) | (draw & 0xFFFF)
            converted.append(seq_val & 0xFFFFFFFF)
            
    elif method == 'polynomial':
        # Polynomial transformation
        for i, draw in enumerate(draws):
            if isinstance(draw, str):
                draw = int(draw)
            # Apply polynomial: ax^3 + bx^2 + cx + d
            poly_val = (draw * draw * draw * 1009) + (draw * draw * 2017) + (draw * 4093) + i
            converted.append(poly_val & 0xFFFFFFFF)
            
    elif method == 'xor_rotate':
        # XOR and rotation method
        for i, draw in enumerate(draws):
            if isinstance(draw, str):
                draw = int(draw)
            # XOR with position, then rotate bits
            xor_val = draw ^ (i * 0x9E3779B9)  # Golden ratio multiplier
            rotated = ((xor_val << 13) | (xor_val >> 19)) & 0xFFFFFFFF
            converted.append(rotated)
    
    else:
        raise ValueError(f"Unknown conversion method: {method}")
    
    return converted

def show_conversion_examples(raw_draws, method):
    """Show examples of the conversion"""
    print(f"\nConversion method: {method}")
    sample = raw_draws[:5]
    converted = convert_lottery_to_32bit(sample, method)
    
    for i, (orig, conv) in enumerate(zip(sample, converted)):
        print(f"  {orig} -> {conv} (0x{conv:08x})")

def analyze_with_method(draws, raw_draws, method):
    """Analyze using specific conversion method"""
    print(f"\n{'='*50}")
    print(f"ANALYSIS USING {method.upper()} METHOD")
    print(f"{'='*50}")
    
    try:
        from advanced_mt_reconstruction import AdvancedMT19937Reconstructor
        reconstructor = AdvancedMT19937Reconstructor()
        
        result = reconstructor.reconstruct_mt_state(draws)
        
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Method: {result.get('method', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 0)*100:.1f}%")
            
            # Check for MT patterns
            likely_mt = result.get('likely_mt', False)
            if likely_mt:
                print(f"⚠️  Shows MT19937-like patterns with {method} conversion")
            else:
                print(f"✓ No MT19937 patterns detected with {method} conversion")
        
        return result
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

def main():
    filename = 'daily3.json'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    # Load raw data
    try:
        with open(filename, 'r') as f:
            lottery_data = json.load(f)
        raw_draws = [entry.get('draw') for entry in lottery_data if entry.get('draw') is not None]
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loaded {len(raw_draws)} lottery draws")
    print(f"Sample: {raw_draws[:10]}")
    print(f"Range: {min(raw_draws)} to {max(raw_draws)}")
    
    # Available conversion methods
    methods = ['hash', 'multiply', 'shift_combine', 'sequence', 'polynomial', 'xor_rotate']
    
    print(f"\nAvailable conversion methods:")
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method}")
    
    # Let user choose or test all
    choice = input(f"\nSelect method (1-{len(methods)}) or 'all' to test all methods: ").strip().lower()
    
    if choice == 'all':
        results = {}
        for method in methods:
            converted = convert_lottery_to_32bit(raw_draws, method)
            show_conversion_examples(raw_draws, method)
            result = analyze_with_method(converted, raw_draws, method)
            if result:
                results[method] = result
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY OF ALL METHODS")
        print(f"{'='*60}")
        for method, result in results.items():
            confidence = result.get('confidence', 0) * 100
            likely_mt = result.get('likely_mt', False)
            print(f"{method:15} | Confidence: {confidence:5.1f}% | MT19937-like: {likely_mt}")
            
    else:
        try:
            method_idx = int(choice) - 1
            if 0 <= method_idx < len(methods):
                method = methods[method_idx]
                converted = convert_lottery_to_32bit(raw_draws, method)
                show_conversion_examples(raw_draws, method)
                analyze_with_method(converted, raw_draws, method)
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

if __name__ == "__main__":
    main()
