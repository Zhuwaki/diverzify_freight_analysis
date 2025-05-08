import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


import logging
logging.basicConfig(level=logging.INFO)


# Get full timestamp as string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Get just the date component as a datetime object
date = datetime.now().strftime("%Y%m%d")


# data cleaning function to standardise the description conversion
# This function will classify the commodity based on the description
def classify_commodity(row):
   # if 'description' not in row or pd.isna(row['description']):
   #     return "UNKNOWN"
    desc = str(row['commodity_description']).strip().lower()

    if desc == 'vinyl':
        return ''.join(filter(str.isalpha, str(row.get('comm_2', ''))))
    elif desc == 'carpet bl':
        return 'Carpet Roll'
    elif desc == 'carpet tile':
        return 'Carpet Tiles'
    elif desc == 'carpet':
        return 'Carpet Roll'
    else:
        return desc


# This function will classify the commodity from old codes to new codes
def map_commodity_group(x):
    x_str = str(x).strip()  # Strip whitespace, just in case

    if x_str == '10':
        return '1CBL'
    elif x_str == '100':
        return '1CPT'
    elif x_str == '40':
        return '1VNL'
    else:
        return x  # Keep original value if none of the above match


# This function will classify the commodity from old codes to new codes and merge with manufacturers


def map_manufacturer(input_df, manufacturer_df=None, base_path="data/input"):
    logging.info("🔧 Running map_manufacturer...")

    # Load manufacturer mapping
    try:
        if manufacturer_df is None:
            manufacturer_path = os.path.join(
                base_path, "Manufacturer List.xlsx")
            manufacturer_df = pd.read_excel(
                manufacturer_path, sheet_name='Sheet1', engine="openpyxl")
    except Exception as e:
        raise Exception(f"💥 Failed to load manufacturer_df: {str(e)}")

    # Normalize column names
    input_df.columns = input_df.columns.str.strip().str.lower().str.replace(" ", "_")
    manufacturer_df.columns = manufacturer_df.columns.str.strip(
    ).str.lower().str.replace(" ", "_")

    # Ensure string type for matching
    input_df['supplier_no'] = input_df['supplier_no'].astype(str)
    manufacturer_df['supplier_no'] = manufacturer_df['supplier_no'].astype(str)

    os.makedirs("data/downloads/cleaning", exist_ok=True)
    filename = f"manufacturer_df{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join("data/downloads/cleaning", filename)
    manufacturer_df.to_csv(filepath, index=False)

    # Merge and map
    merged_df = input_df.merge(
        manufacturer_df[['supplier_no', 'supplier_name']],
        on='supplier_no',
        how='left',
        suffixes=('', '_mapped')
    )

    # Flag matches
    merged_df['match_supplier'] = merged_df['supplier_name_mapped'].apply(
        lambda x: 'Supplier registered' if pd.notna(x) else 'No supplier found'
    )

    # Raise warning if unmatched
    unmatched = merged_df[merged_df['match_supplier'] == 'No supplier found']
    if not unmatched.empty:
        logging.warning(f"⚠️ {len(unmatched)} unmatched supplier(s) found.")
        unmatched.to_csv(
            "data/downloads/cleaning/unmatched_suppliers.csv", index=False)

    logging.info("✅ Mapping manufacturers complete.")
    return merged_df


def map_commodity(input_df, commodity_df=None, base_path="data/input"):
    logging.info("🔧 Running map_commodity...")

    # Load commodity mapping
    try:
        if commodity_df is None:
            commodity_path = os.path.join(
                base_path, "IFS Cloud Commodity Groups.xlsx")
            commodity_df = pd.read_excel(
                commodity_path, sheet_name='Commodity Groups', engine="openpyxl")
    except Exception as e:
        raise Exception(f"💥 Failed to load commodity_df: {str(e)}")

    # Normalize column names
    input_df.columns = input_df.columns.str.strip().str.lower().str.replace(" ", "_")
    commodity_df.columns = commodity_df.columns.str.strip(
    ).str.lower().str.replace(" ", "_")

    # Ensure string type for matching
    input_df['comm_1'] = input_df['comm_1'].astype(str)
    commodity_df['comm_1'] = commodity_df['commodity_group'].astype(str)

    # Save for inspection (optional)
    os.makedirs("data/downloads/cleaning", exist_ok=True)
    filename = f"commodity_df{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join("data/downloads/cleaning", filename)
    commodity_df.to_csv(filepath, index=False)

    # Merge and map
    commodity_df_renamed = commodity_df[['comm_1', 'commodity_group', 'commodity_description', 'commodity_code', 'priority_commodity']].rename(
        columns={'commodity_group': 'commodity_group_mapped'}
    )

    merged_df = input_df.merge(
        commodity_df_renamed,
        on='comm_1',
        how='left'
    )

    logging.warning(merged_df.columns)

    # Flag matches
    merged_df['match_commodity'] = merged_df['commodity_group_mapped'].apply(
        lambda x: 'Commodity Found' if pd.notna(x) else 'Commodity Not Found'
    )

    # Raise warning if unmatched
    unmatched = merged_df[merged_df['match_commodity']
                          == 'Commodity Not Found']
    if not unmatched.empty:
        logging.warning(f"⚠️ {len(unmatched)} unmatched commodity(ies) found.")
        unmatched.to_csv(
            "data/downloads/cleaning/unmatched_commodities.csv", index=False)

    logging.info("✅ Mapping commodity complete.")
    return merged_df


def classify_line_uom(df):
    """
    Standardizes UOM and classifies each line item as Classified or Unclassified based on SQFT/SQYD.
    Vectorized version.
    """
    logging.info("🔧 Running classify_line_uom (vectorized)...")

    # Standardize UOM format
    df['inv_uom'] = df['inv_uom'].replace({'SF': 'SQFT', 'SY': 'SQYD'})
    df['inv_uom'] = df['inv_uom'].str.strip().str.upper()

    # Vectorized line-level classification
    valid_uoms = ['SQFT', 'SQYD']
    df['is_classified'] = df['inv_uom'].isin(valid_uoms)
    df['line_classification'] = np.where(
        df['is_classified'], 'Classified', 'Unclassified')

    logging.info("✅ classify_line_uom (vectorized) complete.")
    return df


def create_conversion_code(input_df):
    """
    Creates a standardized conversion code based on commodity description, group, and UOM.
    """
    logging.info("🔧 Running create_conversion_code...")
    # Normalize column names

    # You may already have classify_commodity and map_commodity_group imported from somewhere else
    input_df['new_commodity_description'] = input_df.apply(
        classify_commodity, axis=1)
    input_df['new_commodity_group'] = input_df['commodity_group_mapped'].apply(
        map_commodity_group)

    input_df['conversion_code'] = (
        input_df['new_commodity_description'].str.replace(
            ' ', '_', regex=True).astype(str)
        + '_' +
        input_df['new_commodity_group'].astype(str)
        + '_' +
        input_df['inv_uom'].astype(str)
    )

    logging.info("✅ create_conversion_code complete.")
    return input_df

# Check if priority products specifcially 1VNL exist in the conversion table


def classify_freight_lines(df):
    """
    Adds:
    - has_freight_line: whether invoice has at least one freight line
    - multiple_freight_lines: whether invoice has multiple freight lines
    """
    logging.info("🔧 Running classify_freight_lines...")

    # Assume 'account' column identifies freight lines (e.g., account == 2008 = freight)
    freight_df = df[df['account'] == 5504]

    freight_summary = freight_df.groupby(
        'invoice_id').size().reset_index(name='freight_line_count')
    freight_summary['has_freight_line'] = freight_summary['freight_line_count'] >= 1
    freight_summary['multiple_freight_lines'] = freight_summary['freight_line_count'] > 1

    df = df.merge(freight_summary[['invoice_id', 'has_freight_line',
                  'multiple_freight_lines']], on='invoice_id', how='left')

    # Fill missing freight info (no freight lines found at all)
    df['has_freight_line'] = df['has_freight_line'].fillna(False)
    df['multiple_freight_lines'] = df['multiple_freight_lines'].fillna(False)

    logging.info("✅ classify_freight_lines complete.")
    return df


def classify_parts_and_commodities(df):
    """
    Adds:
    - multiple_parts: whether invoice has multiple part numbers
    - multiple_commodities: whether invoice has multiple commodity groups
    """
    logging.info("🔧 Running classify_parts_and_commodities...")

    part_summary = df.groupby('invoice_id')[
        'part_no'].nunique().reset_index(name='part_count')
    commodity_summary = df.groupby('invoice_id')[
        'new_commodity_group'].nunique().reset_index(name='commodity_count')

    summary = part_summary.merge(commodity_summary, on='invoice_id')
    summary['multiple_parts'] = summary['part_count'] > 1
    summary['multiple_commodities'] = summary['commodity_count'] > 1

    df = df.merge(summary[['invoice_id', 'multiple_parts',
                  'multiple_commodities']], on='invoice_id', how='left')

    logging.info("✅ classify_parts_and_commodities complete.")
    return df


def classify_priority_commodities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - priority_multiple_commodities: whether invoice has multiple priority commodity groups (only 1CBL, 1VNL, 1CPT)
    """
    logging.info("🔧 Running classify_priority_commodities...")

    # Define priority commodities
    PRIORITY_COMMODITIES = ['1CBL', '1VNL', '1CPT']

    # Filter only priority commodity lines
    priority_df = df[df['new_commodity_group'].isin(PRIORITY_COMMODITIES)]

    # Group and summarize
    commodity_summary = priority_df.groupby('invoice_id')[
        'new_commodity_group'].nunique().reset_index(name='priority_commodity_count')

    # Flagging
    commodity_summary['priority_multiple_commodities'] = commodity_summary['priority_commodity_count'] > 1

    # Merge back into the main DataFrame
    df = df.merge(commodity_summary[['invoice_id', 'priority_multiple_commodities']],
                  on='invoice_id', how='left')

    logging.info("✅ classify_priority_commodities complete.")
    return df


def classify_priority_products_2008(df):
    """
    Adds:
    - all_invoice_priority_products_2008: whether all 2008 lines are priority commodities
    - any_invoice_priority_products_2008: whether any 2008 line is a priority commodity
    """
    logging.info("🔧 Running classify_priority_products_2008...")

    priority_commodities = ['1CBL', '1CPT', '1VNL']

    # Only account 2008 rows
    account_2008_df = df[df['account'] == 2008]

    def check_all_priority(group):
        return all(x in priority_commodities for x in group['new_commodity_group'])

    def check_any_priority(group):
        return any(x in priority_commodities for x in group['new_commodity_group'])

    priority_status = account_2008_df.groupby('invoice_id').apply(
        lambda group: pd.Series({
            'all_invoice_priority_products_2008': check_all_priority(group),
            'any_invoice_priority_products_2008': check_any_priority(group)
        })
    ).reset_index()

    df = df.merge(priority_status, on='invoice_id', how='left')

    # Fill NaN with False (if invoice had no account 2008 rows)
    df['all_invoice_priority_products_2008'] = df['all_invoice_priority_products_2008'].fillna(
        False)
    df['any_invoice_priority_products_2008'] = df['any_invoice_priority_products_2008'].fillna(
        False)

    logging.info("✅ classify_priority_products_2008 complete.")
    return df


def add_freight_per_invoice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'freight_per_invoice' column to the DataFrame where each row reflects
    the total freight cost (ACCOUNT == 5504) for its invoice_id.

    Parameters:
    - df: DataFrame with at least 'invoice_id', 'account', and 'invoice_line_total' columns

    Returns:
    - df: updated DataFrame with 'freight_per_invoice' column
    """
    logging.info("✅ Calculating freight per invoice.")
    # Step 1: Filter freight lines
    freight_lines = df[df['account'] == 5504]

    # Step 2: Sum freight per invoice
    freight_per_invoice = (
        freight_lines
        .groupby('invoice_id', as_index=False)['invoice_line_total']
        .sum()
        .rename(columns={'invoice_line_total': 'freight_per_invoice'})
    )

    # Step 3: Merge and propagate to all rows
    df = df.merge(freight_per_invoice, on='invoice_id', how='left')

    # Step 4: Fill NaN with 0 (invoices with no freight)
    df['freight_per_invoice'] = df['freight_per_invoice'].fillna(0)
    logging.info("✅ Completed adding freight to invoice.")

    return df


def add_invoice_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'invoice_total' column to the DataFrame where each row reflects
    the total invoice cost (sum of all line items) for its invoice_id.

    Parameters:
    - df: DataFrame with at least 'invoice_id', 'invoice_line_total' columns

    Returns:
    - df: updated DataFrame with 'invoice_total' column
    """
    logging.info("✅ Calculating invoice total.")
    # Step 1: Sum invoice line totals per invoice
    invoice_totals = (
        df
        .groupby('invoice_id', as_index=False)['invoice_line_total']
        .sum()
        .rename(columns={'invoice_line_total': 'invoice_total'})
    )

    # Step 2: Merge and propagate to all rows
    df = df.merge(invoice_totals, on='invoice_id', how='left')

    # Step 3: Fill NaN with 0 (invoices with no lines)
    df['invoice_total'] = df['invoice_total'].fillna(0)
    logging.info("✅ Completed adding invoice total.")

    return df


def priority_product_composition(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("✅ Estimating priority product composition.")

    # Priority yes totals
    priority_yes_totals = df[
        (df['account'] == 2008) & (df['priority_commodity'] == 'Yes')
    ].groupby('invoice_id')['invoice_line_total'].sum().reset_index()
    priority_yes_totals.rename(
        columns={'invoice_line_total': 'priority_product_total'}, inplace=True)

    # Priority no totals
    priority_no_totals = df[
        (df['account'] == 2008) & (df['priority'] == 'No')
    ].groupby('invoice_id')['invoice_line_total'].sum().reset_index()
    priority_no_totals.rename(
        columns={'invoice_line_total': 'non_priority_product_total'}, inplace=True)

    # Merge totals
    priority_totals = pd.merge(
        priority_yes_totals, priority_no_totals, on='invoice_id', how='outer').fillna(0)

    # Calculate invoice total
    priority_totals['invoice_product_total'] = (
        priority_totals['priority_product_total'] +
        priority_totals['non_priority_product_total']
    )

    # 🛡️ Safe calculation: if invoice_product_total == 0, set percentages to 0
    priority_totals['percentage_priority'] = np.where(
        priority_totals['invoice_product_total'] == 0,
        0,
        (priority_totals['priority_product_total'] /
         priority_totals['invoice_product_total']) * 100
    )

    priority_totals['percentage_non_priority'] = np.where(
        priority_totals['invoice_product_total'] == 0,
        0,
        (priority_totals['non_priority_product_total'] /
         priority_totals['invoice_product_total']) * 100
    )

    # 🛡️ Safe creation of boolean flag
    priority_totals['pct_priority_greater_than_70'] = priority_totals['percentage_priority'] > 70

    # 🛡️ Ensure no NaNs in pct_priority_greater_than_70
    priority_totals['pct_priority_greater_than_70'] = priority_totals['pct_priority_greater_than_70'].fillna(
        False).astype(bool)

    # Merge back
    df = df.merge(
        priority_totals[['invoice_id', 'priority_product_total', 'non_priority_product_total',
                         'percentage_priority', 'percentage_non_priority', 'pct_priority_greater_than_70']],
        on='invoice_id',
        how='left'
    )
    # 🛡️ After merge, fill missing values
    df['priority_product_total'] = df['priority_product_total'].fillna(0)
    df['non_priority_product_total'] = df['non_priority_product_total'].fillna(
        0)
    df['percentage_priority'] = df['percentage_priority'].fillna(0)
    df['percentage_non_priority'] = df['percentage_non_priority'].fillna(0)
    df['pct_priority_greater_than_70'] = df['pct_priority_greater_than_70'].fillna(
        False).astype(bool)
    # 🧪 Diagnostic check: is there any invalid float still leaking?
    invalids = df[
        df[['percentage_priority', 'percentage_non_priority']].isnull().any(axis=1) |
        df[['percentage_priority', 'percentage_non_priority']].isin(
            [np.inf, -np.inf]).any(axis=1)
    ]

    invalids.to_csv('data/downloads/invalid.csv', index=False)

    if not invalids.empty:
        print("⚠️ Found invalid entries after merge!")
        print(invalids)
    else:
        print("✅ No invalid entries detected.")

    logging.info("✅ completed increasing sample size.")
    return df


def filter_valid_invoices(mapped_df):
    logging.info("✅ Filtering valid invoices.")

    # Apply the filters
    filtered_df = mapped_df[
        (mapped_df['has_freight_line'] == True) &
        (mapped_df['invoiced_line_qty'] > 0) &
        (mapped_df['freight_per_invoice'] > 0)
    ]
    filtered_df = filtered_df[filtered_df['conversion_code'] != 'nan_nan_nan']

    return filtered_df


def filter_sample_invoices(mapped_df):
    logging.info("✅ Filtering sample invoices.")
    site_list = [
        "DIT", "SPW", "SPN", "SPCP", "SPT",
        "PVF", "SPHU", "SPTM", "FSU", "CTS", "SPJ",
    ]

    # Apply the filters
    filtered_df = mapped_df[
        (mapped_df['any_invoice_priority_products_2008'] == True) &
        (mapped_df['has_freight_line'] == True) &
        (mapped_df['site'].isin(site_list)) &
        (mapped_df['invoiced_line_qty'] > 0) &
        (mapped_df['freight_per_invoice'] > 0) &
        (mapped_df['pct_priority_greater_than_70'] == True)
    ]
    filtered_df = filtered_df[filtered_df['conversion_code'] != 'nan_nan_nan']

    return filtered_df


# BACK UP FUNCTIONS
def classify_invoice_priority_uom(df):
    """
    Flags whether all priority commodity lines (1CBL/1CPT/1VNL) in each invoice are classified.
    Adds:
    - invoice_priority_classified: True if all relevant lines are classified, else False
    """
    logging.info("🔧 Running classify_invoice_priority_uom...")

    # Only priority commodities considered
    priority_commodities = ['1CBL', '1CPT', '1VNL']
    priority_df = df[df['new_commodity_group'].isin(priority_commodities)]

    # Group by invoice and check if all priority lines are classified
    invoice_classification = priority_df.groupby(
        'invoice_id')['is_classified'].all().reset_index()
    invoice_classification.rename(
        columns={'is_classified': 'invoice_priority_classified'}, inplace=True)

    # Merge back into main dataframe
    df = df.merge(invoice_classification, on='invoice_id', how='left')
    df['all_invoice_priority_commodities'] = df['invoice_priority_classified'].fillna(
        "NO_PRIORITY")

    logging.info("✅ classify_invoice_priority_uom complete.")
    return df


def classify_invoice_priority_conversion(df, conversion_lookup):
    """
    Flags whether all priority commodity lines (1CBL/1CPT/1VNL) in each invoice can be converted.
    - 1CBL and 1CPT: Automatically convertible (no lookup)
    - 1VNL: Must have valid conversion_code in lookup
    Adds:
    - can_be_converted (at line level)
    - invoice_priority_conversion_success (at invoice level)
    """
    logging.info("🔧 Running classify_invoice_priority_conversion...")

    # Priority commodities
    priority_commodities = ['1CBL', '1CPT', '1VNL']

    # Step 1: Filter to priority commodities only
    priority_df = df[df['new_commodity_group'].isin(
        priority_commodities)].copy()

    # Step 2: Line-level conversion feasibility
    def check_conversion(row):
        try:
            group = row['new_commodity_group']
            if group in ['1CBL', '1CPT']:
                return True
            elif group == '1VNL':
                return row['conversion_code'] in conversion_lookup
            else:
                return False
        except Exception as e:
            print("⚠️ Row causing error:", row.to_dict())
            raise e

    priority_df['can_be_converted'] = priority_df.apply(
        check_conversion, axis=1)

    # Step 3: Invoice-level aggregation
    invoice_conversion_status = priority_df.groupby(
        'invoice_id')['can_be_converted'].all().reset_index()
    invoice_conversion_status.rename(
        columns={'can_be_converted': 'invoice_priority_conversion_success'}, inplace=True)

    # Step 4: Merge back into main dataframe
    df = df.merge(invoice_conversion_status, on='invoice_id', how='left')

    logging.info("✅ classify_invoice_priority_conversion complete.")
    return df
