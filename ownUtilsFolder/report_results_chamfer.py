import argparse
from pathlib import Path
import re
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows


def parse_chamfer_file(file_path):
    """
    Parse a chamfer.txt file and extract all metrics.
    
    Args:
        file_path: Path to the chamfer.txt file
        
    Returns:
        dict: Extracted metrics and metadata
    """
    data = {
        'file_path': str(file_path),
        'parent_folder': file_path.parent.name,
        'full_path': str(file_path.parent),
    }
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract point counts
    ground_match = re.search(r'Ground truth points:\s*([0-9,]+)', content)
    recon_match = re.search(r'Reconstruction points:\s*([0-9,]+)', content)
    
    if ground_match:
        data['ground_truth_points'] = int(ground_match.group(1).replace(',', ''))
    if recon_match:
        data['reconstruction_points'] = int(recon_match.group(1).replace(',', ''))
    
    # Extract one-sided Chamfer distances
    g2r_match = re.search(r'Ground → Reconstruction:\s*([0-9.e+-]+)', content)
    r2g_match = re.search(r'Reconstruction → Ground:\s*([0-9.e+-]+)', content)
    
    if g2r_match:
        data['cd_ground_to_recon'] = float(g2r_match.group(1))
    if r2g_match:
        data['cd_recon_to_ground'] = float(r2g_match.group(1))
    
    # Extract symmetric Chamfer distance
    sym_match = re.search(r'Symmetric CD:\s*([0-9.e+-]+)', content)
    if sym_match:
        data['cd_symmetric'] = float(sym_match.group(1))
    
    # Extract statistics for Ground → Reconstruction
    section = re.search(
        r'Detailed Statistics \(Ground → Reconstruction\):(.*?)(?=\n\nDetailed Statistics|\Z)',
        content, re.DOTALL
    )
    
    if section:
        stats_text = section.group(1)
        data['g2r_mean'] = extract_float(stats_text, r'mean:\s*([0-9.e+-]+)')
        data['g2r_rmse'] = extract_float(stats_text, r'rmse:\s*([0-9.e+-]+)')
        data['g2r_median'] = extract_float(stats_text, r'median:\s*([0-9.e+-]+)')
        data['g2r_std'] = extract_float(stats_text, r'std:\s*([0-9.e+-]+)')
        data['g2r_min'] = extract_float(stats_text, r'min:\s*([0-9.e+-]+)')
        data['g2r_max'] = extract_float(stats_text, r'max:\s*([0-9.e+-]+)')
        data['g2r_p95'] = extract_float(stats_text, r'p95:\s*([0-9.e+-]+)')
        data['g2r_p99'] = extract_float(stats_text, r'p99:\s*([0-9.e+-]+)')
    
    # Extract statistics for Reconstruction → Ground
    section = re.search(
        r'Detailed Statistics \(Reconstruction → Ground\):(.*)',
        content, re.DOTALL
    )
    
    if section:
        stats_text = section.group(1)
        data['r2g_mean'] = extract_float(stats_text, r'mean:\s*([0-9.e+-]+)')
        data['r2g_rmse'] = extract_float(stats_text, r'rmse:\s*([0-9.e+-]+)')
        data['r2g_median'] = extract_float(stats_text, r'median:\s*([0-9.e+-]+)')
        data['r2g_std'] = extract_float(stats_text, r'std:\s*([0-9.e+-]+)')
        data['r2g_min'] = extract_float(stats_text, r'min:\s*([0-9.e+-]+)')
        data['r2g_max'] = extract_float(stats_text, r'max:\s*([0-9.e+-]+)')
        data['r2g_p95'] = extract_float(stats_text, r'p95:\s*([0-9.e+-]+)')
        data['r2g_p99'] = extract_float(stats_text, r'p99:\s*([0-9.e+-]+)')
    
    return data


def extract_float(text, pattern):
    """Helper function to extract float from text using regex."""
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def find_chamfer_files(root_dir):
    """
    Recursively find all chamfer.txt files in directory.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        list: Paths to all chamfer.txt files found
    """
    root_path = Path(root_dir)
    chamfer_files = list(root_path.rglob('chamfer.txt'))
    return sorted(chamfer_files)


def create_excel_report(data_list, output_path):
    """
    Create an Excel file with all chamfer distance results.
    
    Args:
        data_list: List of dictionaries containing parsed data
        output_path: Path to save the Excel file
    """
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Reorder columns for better readability
    column_order = [
        'parent_folder',
        'full_path',
        'ground_truth_points',
        'reconstruction_points',
        'cd_symmetric',
        'cd_ground_to_recon',
        'cd_recon_to_ground',
        'g2r_mean', 'g2r_rmse', 'g2r_median', 'g2r_std', 
        'g2r_min', 'g2r_max', 'g2r_p95', 'g2r_p99',
        'r2g_mean', 'r2g_rmse', 'r2g_median', 'r2g_std',
        'r2g_min', 'r2g_max', 'r2g_p95', 'r2g_p99',
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Rename columns for better display
    df = df.rename(columns={
        'parent_folder': 'Folder',
        'full_path': 'Full Path',
        'ground_truth_points': 'Ground Truth Points',
        'reconstruction_points': 'Reconstruction Points',
        'cd_symmetric': 'Symmetric CD',
        'cd_ground_to_recon': 'CD: Ground→Recon',
        'cd_recon_to_ground': 'CD: Recon→Ground',
        'g2r_mean': 'G2R: Mean',
        'g2r_rmse': 'G2R: RMSE',
        'g2r_median': 'G2R: Median',
        'g2r_std': 'G2R: Std Dev',
        'g2r_min': 'G2R: Min',
        'g2r_max': 'G2R: Max',
        'g2r_p95': 'G2R: P95',
        'g2r_p99': 'G2R: P99',
        'r2g_mean': 'R2G: Mean',
        'r2g_rmse': 'R2G: RMSE',
        'r2g_median': 'R2G: Median',
        'r2g_std': 'R2G: Std Dev',
        'r2g_min': 'R2G: Min',
        'r2g_max': 'R2G: Max',
        'r2g_p95': 'R2G: P95',
        'r2g_p99': 'R2G: P99',
    })
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Chamfer Distances', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Chamfer Distances']
        
        # Style the header row
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')
        
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Freeze the header row
        worksheet.freeze_panes = 'A2'
        
        # Add summary statistics sheet
        if len(df) > 0:
            summary_data = {
                'Metric': [],
                'Mean': [],
                'Median': [],
                'Min': [],
                'Max': [],
                'Std Dev': []
            }
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in numeric_cols:
                summary_data['Metric'].append(col)
                summary_data['Mean'].append(df[col].mean())
                summary_data['Median'].append(df[col].median())
                summary_data['Min'].append(df[col].min())
                summary_data['Max'].append(df[col].max())
                summary_data['Std Dev'].append(df[col].std())
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
            
            # Style summary sheet
            summary_ws = writer.sheets['Summary Statistics']
            for cell in summary_ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
            
            for column in summary_ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                summary_ws.column_dimensions[column_letter].width = adjusted_width


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate chamfer.txt files into an Excel spreadsheet'
    )
    parser.add_argument('--directory', default='.',
                       help='Root directory to search for chamfer.txt files (default: current directory)')
    parser.add_argument('--output', default=None,
                       help='Output Excel file path (default: chamfer.xlsx in search directory)')
    
    args = parser.parse_args()
    
    root_dir = Path(args.directory)
    
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    print(f"Searching for chamfer.txt files in: {root_dir}")
    chamfer_files = find_chamfer_files(root_dir)
    
    if not chamfer_files:
        print("No chamfer.txt files found!")
        return
    
    print(f"Found {len(chamfer_files)} chamfer.txt file(s)")
    
    # Parse all files
    data_list = []
    for i, file_path in enumerate(chamfer_files, 1):
        print(f"  [{i}/{len(chamfer_files)}] Parsing: {file_path}")
        try:
            data = parse_chamfer_file(file_path)
            data_list.append(data)
        except Exception as e:
            print(f"    Warning: Failed to parse {file_path}: {e}")
    
    if not data_list:
        print("No valid data extracted from files!")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = root_dir / 'chamfer.xlsx'
    
    print(f"\nCreating Excel report: {output_path}")
    create_excel_report(data_list, output_path)
    
    print(f"\nSuccess! Consolidated {len(data_list)} results into {output_path}")
    print(f"  - Main results: 'Chamfer Distances' sheet")
    print(f"  - Summary statistics: 'Summary Statistics' sheet")


if __name__ == "__main__":
    main()