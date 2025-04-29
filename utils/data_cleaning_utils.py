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

APPLY_MARKET_DISCOUNT = True  # Set to True to apply the discount
# === Configurable discount factor for market comparison ===
MARKET_RATE_DISCOUNT = 0.30  # 30% reduction


# data cleaning function to standardise the description conversion
# This function will classify the commodity based on the description
def classify_commodity(row):
    desc = str(row['description']).strip()
    desc_lower = desc.lower()

    if desc_lower == 'vinyl':
        return ''.join(filter(str.isalpha, str(row['comm_2'])))
    elif desc_lower == 'carpet bl':
        return 'Carpet Roll'
    elif desc_lower == 'carpet tile':
        return 'Carpet Tiles'
    elif desc_lower == 'carpet':
        return 'Carpet Roll'
    else:
        return desc  # Default fallback to original


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


def map_commodity_and_manufacturer(input_df, commodity_df=None, manufacturer_df=None, base_path="data/input"):
    """
    Maps commodity and manufacturer information onto the input dataframe.
    """
    logging.info("🔧 Running map_commodity_and_manufacturer...")

    # Load commodity mapping
    try:
        if commodity_df is None:
            commodity_path = os.path.join(
                base_path, "IFS Cloud Commodity Groups.xlsx")
            commodity_df = pd.read_excel(
                commodity_path, sheet_name='Commodity Groups', engine="openpyxl")
    except Exception as e:
        raise Exception(f"💥 Failed to load commodity_df: {str(e)}")

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
    commodity_df.columns = commodity_df.columns.str.strip(
    ).str.lower().str.replace(" ", "_")
    manufacturer_df.columns = manufacturer_df.columns.str.strip(
    ).str.lower().str.replace(" ", "_")

    # Merge commodity mapping
    commodity_df['comm_1'] = commodity_df['commodity_group'].astype(str)
    input_df['comm_1'] = input_df['comm_1'].astype(str)
    input_df = input_df.merge(commodity_df, on='comm_1', how='left')
    input_df['match_commodity'] = input_df['commodity_group'].apply(
        lambda x: 'Commodity Found' if pd.notna(x) else 'Commodity Not Found'
    )

    # Merge manufacturer mapping
    manufacturer_df['supplier_no'] = manufacturer_df['supplier_no'].astype(str)
    input_df['supplier_no'] = input_df['supplier_no'].astype(str)
    input_df = input_df.merge(
        manufacturer_df[['supplier_no']], on='supplier_no', how='left')
    input_df['match_supplier'] = input_df['supplier_name'].apply(
        lambda x: 'Supplier registered' if pd.notna(x) else 'No supplier found'
    )

    logging.info("✅ map_commodity_and_manufacturer complete.")
    return input_df


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
    df['invoice_priority_classified'] = df['invoice_priority_classified'].fillna(
        "NO_PRIORITY")

    logging.info("✅ classify_invoice_priority_uom complete.")
    return df


def create_conversion_code(input_df):
    """
    Creates a standardized conversion code based on commodity description, group, and UOM.
    """
    logging.info("🔧 Running create_conversion_code...")

    # You may already have classify_commodity and map_commodity_group imported from somewhere else
    input_df['new_commodity_description'] = input_df.apply(
        classify_commodity, axis=1)
    input_df['new_commodity_group'] = input_df['commodity_group'].apply(
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
        group = row['new_commodity_group']
        if group in ['1CBL', '1CPT']:
            return True  # Automatically convertible
        elif group == '1VNL':
            return row['conversion_code'] in conversion_lookup
        else:
            return False  # Should not happen but for safety

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


def priority_product_composition(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("✅ increasing freight per invoice.")

    # Priority yes totals
    priority_yes_totals = df[
        (df['account'] == 2008) & (df['priority'] == 'Yes')
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


def resampling(df: pd.DataFrame):
    # Step 1: Filter only account 2008
    df_2008 = df[df['account'] == 2008].copy()

    # Step 2: Total for priority == Yes per invoice
    priority_yes_totals = df_2008[df_2008['priority'] == 'Yes'].groupby(
        'invoice_id')['invoice_line_total'].sum().reset_index()
    priority_yes_totals.rename(
        columns={'invoice_line_total': 'priority_yes_total'}, inplace=True)

    # Step 3: Total for all account 2008 per invoice
    total_2008_totals = df_2008.groupby(
        'invoice_id')['invoice_line_total'].sum().reset_index()
    total_2008_totals.rename(
        columns={'invoice_line_total': 'total_2008_invoice_total'}, inplace=True)

    # Step 4: Merge the two summaries
    merged_totals = pd.merge(
        total_2008_totals, priority_yes_totals, on='invoice_id', how='left')
    merged_totals['priority_yes_total'] = merged_totals['priority_yes_total'].fillna(
        0)

    # Step 5: Calculate percentage
    merged_totals['pct_priority_yes_2008'] = (
        merged_totals['priority_yes_total'] /
        merged_totals['total_2008_invoice_total'] * 100
    )
    # ✅ Add flag column based on percentage threshold
    merged_totals['baseline_sample'] = merged_totals['pct_priority_yes_2008'] > 70
    # Step 6: Merge back into original DataFrame
    df = df.merge(merged_totals, on='invoice_id', how='left')

    return df


def filter_valid_invoices(mapped_df):

    # Apply the filters
    filtered_df = mapped_df[
        (mapped_df['has_freight_line'] == True) &
        (mapped_df['invoiced_line_qty'] > 0) &
        (mapped_df['freight_per_invoice'] > 0)
    ]
    filtered_df = filtered_df[filtered_df['conversion_code'] != 'nan_nan_nan']

    return filtered_df


def filter_sample_invoices(mapped_df):
    site_list = ['DIT', 'SPJ', 'SPN', 'SPT', 'SPW',
                 'SPCP', 'SPHU', 'KUS', 'PVF', 'SPTM']

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


# Backup functions


def invoice_uom_classification(df):
    logging.info("✅ fixing unit of measure.")

    # Step 1: Identify invoice_ids where ALL rows with ACCOUNT == 2008 are classified
    uom_output = df
    classified_invoice_ids = (
        uom_output[uom_output['account'] == 2008]
        .groupby('invoice_id')['line_classification']
        .apply(lambda x: all(x == 'Classified'))
    )

    # Step 2: Filter to only invoice IDs where ALL 2008 accounts are classified
    fully_classified_ids = classified_invoice_ids[classified_invoice_ids].index

    # Step 3: Create a new column to mark if entire invoice is considered classified (based on the 2008 rule)
    uom_output['all_invoice_commodity_uom_classified'] = uom_output['invoice_id'].isin(
        fully_classified_ids)

    return uom_output


def flag_fully_converted_invoices(df: pd.DataFrame, conversion_csv_path: str) -> pd.DataFrame:
    """
    Flags invoices where all account == 2008 rows have valid conversion codes.

    Parameters:
    - df: main DataFrame with invoice lines
    - conversion_csv_path: path to the CSV file with valid conversion codes

    Returns:
    - df: updated DataFrame with a boolean column 'all_2008_accounts_converted'
    """

    logging.info("✅ Flagging fully converted invoices.")
    # Load and prepare conversion table
    rates_df = pd.read_csv(conversion_csv_path)
    rates_df['conversion_code'] = rates_df['conversion_code'].astype(str)
    df['conversion_code'] = df['conversion_code'].astype(str)

    # Set of valid codes
    valid_codes = set(rates_df['conversion_code'].unique())

    # Filter 2008 account rows
    df_2008 = df[df['account'] == 2008].copy()  # assuming column is lowercase

    # Check validity per invoice
    invoice_validity = df_2008.groupby('invoice_id')['conversion_code'].apply(
        lambda codes: all(code in valid_codes for code in codes)
    )

    # Flag full matches
    fully_valid_invoice_ids = invoice_validity[invoice_validity].index
    df['all_2008_accounts_converted'] = df['invoice_id'].isin(
        fully_valid_invoice_ids)

    # Optional logging or return of count
    count_all_valid_invoices = df[df['all_2008_accounts_converted']
                                  ]['invoice_id'].nunique()
    print(f"✅ {count_all_valid_invoices} invoices have all account == 2008 rows with valid conversion codes")

    return df


def enrich_invoice_flags(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("✅ Enriching invoice data.")
    # Step 1: Flag invoices with at least one freight line (ACCOUNT == 5504)
    freight_invoice_ids = df[df['account'] == 5504]['invoice_id'].unique()
    df['has_freight_line'] = df['invoice_id'].isin(freight_invoice_ids)
    count_freight_invoices = df[df['has_freight_line']]['invoice_id'].nunique()
    print(
        f"Number of invoices with at least one freight line: {count_freight_invoices}")

    # Step 2: Flag invoices with multiple freight lines
    freight_count = df[df['account'] == 5504].groupby('invoice_id').size()
    df['multiple_freight_lines'] = df['invoice_id'].map(
        freight_count > 1).fillna(False)
    count_multiple_freight_invoices = df[df['multiple_freight_lines']]['invoice_id'].nunique(
    )
    print(
        f"Number of invoices with multiple freight lines: {count_multiple_freight_invoices}")

    # Step 3: Flag invoices with multiple distinct PART NO (ACCOUNT == 2008)
    df_2008 = df[df['account'] == 2008]
    component_count = df_2008.groupby('invoice_id')['part_no'].nunique()
    df['multiple_parts'] = df['invoice_id'].map(
        component_count > 1).fillna(False)
    count_multiple_parts_invoices = df[df['multiple_parts']]['invoice_id'].nunique(
    )
    print(
        f"Number of invoices with multiple distinct parts: {count_multiple_parts_invoices}")

    # Step 4: Flag invoices with multiple distinct COMMODITY GROUP (ACCOUNT == 2008)
    commodity_count = df_2008.groupby(
        'invoice_id')['new_commodity_group'].nunique()
    df['multiple_commodities'] = df['invoice_id'].map(
        commodity_count > 1).fillna(False)
    count_multiple_commodities_invoices = df[df['multiple_commodities']]['invoice_id'].nunique(
    )
    print(
        f"Number of invoices with multiple distinct commodities: {count_multiple_commodities_invoices}")

    # Step 5: Flag invoices where all ACCOUNT == 2008 rows have Priority == 'Yes'
    priority_flag_all = df_2008.groupby(
        'invoice_id')['priority'].apply(lambda x: all(x == 'Yes'))
    priority_invoice_ids_all = priority_flag_all[priority_flag_all].index
    df['all__invoice_priority_products_(2008)'] = df['invoice_id'].isin(
        priority_invoice_ids_all)
    count_priority_invoices = df[df['all__invoice_priority_products_(2008)']]['invoice_id'].nunique(
    )
    print(
        f"Number of invoices where all ACCOUNT == 2008 have Priority == 'Yes': {count_priority_invoices}")

    # Step 6: Flag invoices where any ACCOUNT == 2008 row has Priority == 'Yes'
    priority_flag_any = df_2008.groupby(
        'invoice_id')['priority'].apply(lambda x: any(x == 'Yes'))
    priority_invoice_ids_any = priority_flag_any[priority_flag_any].index
    df['any__invoice_priority_products_(2008)'] = df['invoice_id'].isin(
        priority_invoice_ids_any)
    count_any_priority_invoices = df[df['any__invoice_priority_products_(2008)']]['invoice_id'].nunique(
    )
    print(
        f"Number of invoices where at least one ACCOUNT == 2008 has Priority == 'Yes': {count_any_priority_invoices}")

    return df
