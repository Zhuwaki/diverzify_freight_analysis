import os
import pandas as pd
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
        (df['account'] == 2008) & (df['priority_commodity'] == 'No')
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

    invalids.to_csv('data/downloads/cleaning/invalid.csv', index=False)

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
        "BSC", "CCS", "CCSG", "DCN", "DIN", "DIT", "DPW", "DSL", "FSC", "FSG",
        "FSNC", "FSU", "KFC", "PSC", "PSLV", "PSP", "PSS", "PSUC", "PVF", "RDW",
        "SPA", "SPB", "SPC", "SPCB", "SPCP", "SPD", "SPHU", "SPHV", "SPJ", "SPK",
        "SPL", "SPLA", "SPLV", "SPN", "SPP", "SPS", "SPSA", "SPT", "SPTG", "SPTM",
        "SPW", "SPWV"
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
   # filtered_df = filtered_df[filtered_df['conversion_code'] != 'nan_nan_nan']

    return filtered_df


def classify_priority_invoice_with_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two invoice-level columns based only on priority product presence:
    - invoice_commodity_group: the group (e.g., 1CBL) of a priority product, if any
    - invoice_commodity_description: the description (e.g., Carpet Roll) of that priority product
    If no priority product exists for an invoice, both fields are set to 'NO_PRIORITY_PRODUCT'.
    """
    logging.info("🔧 Running classify_priority_invoice_with_labels...")

    PRIORITY_GROUPS = ['1CBL', '1CPT', '1VNL']

    # Filter only lines that are priority products
    priority_df = df[df['new_commodity_group'].isin(PRIORITY_GROUPS)]

    # For each invoice, get the first priority product's group and description
    priority_info = priority_df.groupby('invoice_id').agg(
        invoice_commodity_group=('new_commodity_group', 'first'),
        invoice_commodity_description=('new_commodity_description', 'first')
    ).reset_index()

    # Merge back into full dataset
    df = df.merge(priority_info, on='invoice_id', how='left')

    # Fill invoices without priority products
    df['invoice_commodity_group'] = df['invoice_commodity_group'].fillna(
        'NO_PRIORITY_PRODUCT')
    df['invoice_commodity_description'] = df['invoice_commodity_description'].fillna(
        'NO_PRIORITY_PRODUCT')

    logging.info("✅ classify_priority_invoice_with_labels complete.")
    return df


def map_supplier_characteristics(input_df: pd.DataFrame, supplier_df: pd.DataFrame = None, base_path: str = "data/input") -> pd.DataFrame:
    """
    Merges supplier metadata on 'supplier_name' and flags if supplier not matched.

    Parameters:
    - input_df: line-level or invoice-level data containing 'supplier_name'
    - supplier_df: optional pre-loaded DataFrame of supplier metadata
    - base_path: directory path to load supplier metadata from if not provided

    Expected columns in supplier_df:
    - supplier_name, freight_spend, commodity_spend, unique_invoice_id,
      unique_site, supplier_mode_master, location, model, supplier_mode
    """
    logging.info("🔧 Running map_supplier_characteristics...")

    # Load supplier metadata if not provided
    try:
        if supplier_df is None:
            supplier_path = os.path.join(base_path, "Supplier Summary.xlsx")
            supplier_df = pd.read_excel(
                supplier_path, sheet_name="Sheet1", engine="openpyxl")
            logging.info(f"📥 Loaded supplier metadata from {supplier_path}")
    except Exception as e:
        raise Exception(f"💥 Failed to load supplier_df: {str(e)}")

    # Standardize for matching
    input_df['supplier_name'] = input_df['supplier_name'].str.strip().str.upper()
    supplier_df['supplier_name'] = supplier_df['supplier_name'].str.strip(
    ).str.upper()

    # Merge
    merged_df = input_df.merge(
        supplier_df,
        on='supplier_name',
        how='left',
        suffixes=('', '_supplier')
    )

    # Match flag
    merged_df['supplier_match_flag'] = np.where(
        # or use another key column from supplier_df
        merged_df['supplier_mode'].notna(),
        'Supplier Matched',
        'No Supplier Match'
    )

    # Export unmatched
    unmatched = merged_df[merged_df['supplier_match_flag']
                          == 'No Supplier Match']
    if not unmatched.empty:
        os.makedirs("data/downloads/cleaning", exist_ok=True)
        unmatched.to_csv(
            "data/downloads/cleaning/unmatched_suppliers_by_name.csv", index=False)
        logging.warning(f"⚠️ {len(unmatched)} unmatched supplier(s) by name.")

    logging.info("✅ map_supplier_characteristics complete.")
    return merged_df
