import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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


def data_cleaning(input_df, commodity_df=None, manufacturer_df=None):
    logging.info("🔧 Running data_cleaning...")
    try:
        if commodity_df is None:
            commodity_df = pd.read_excel(
                "data/input/IFS Cloud Commodity Groups.xlsx", sheet_name='Commodity Groups', engine="openpyxl")
    except Exception as e:
        raise Exception(f"💥 Failed to load commodity_df: {str(e)}")

    try:
        if manufacturer_df is None:
            manufacturer_df = pd.read_excel(
                "data/input/Manufacturer List.xlsx", sheet_name='Sheet1', engine="openpyxl")
    except Exception as e:
        raise Exception(f"💥 Failed to load manufacturer_df: {str(e)}")
    # print(manufacturer_df)
    # print(commodity_df)
    # Normalize column names to lowercase and replace spaces with underscores
    # Normalize column names to lowercase and replace spaces with underscores
    input_df.columns = input_df.columns.str.strip().str.lower().str.replace(" ", "_")
    commodity_df.columns = commodity_df.columns.str.strip(
    ).str.lower().str.replace(" ", "_")
    manufacturer_df.columns = manufacturer_df.columns.str.strip(
    ).str.lower().str.replace(" ", "_")
    # Convert 'Commodity Group' to string and create a new column 'COMM 1'
    commodity_df['comm_1'] = commodity_df['commodity_group'].astype(str)
    # Convert 'Commodity Group' to string in the main DataFrame
    input_df['comm_1'] = input_df['comm_1'].astype(str)
    # Perform the join on the 'COMM 1' column
    input_commodity_df = input_df.merge(commodity_df, on='comm_1', how='left')
# Flag matched and unmatched rows clearly
    input_commodity_df['match_commodity'] = input_commodity_df['commodity_group'].apply(
        lambda x: 'Commodity Found' if pd.notna(x) else 'Commodity Not Found'
    )
    # Replace values in the 'uom' column
    input_commodity_df['inv_uom'] = input_commodity_df['inv_uom'].replace(
        {'SF': 'SQFT', 'SY': 'SQYD'})

    # Convert 'Commodity Group' to string and create a new column 'COMM 1'
    manufacturer_df['supplier_no'] = manufacturer_df['supplier_no'].astype(str)
    # Convert 'Commodity Group' to string in the main DataFrame
    input_commodity_df['supplier_no'] = input_commodity_df['supplier_no'].astype(
        str)
    # Perform the join on the 'COMM 1' column
    input_commodity_manufactuer_df = input_commodity_df.merge(
        manufacturer_df[['supplier_no']], on='supplier_no', how='left')
    input_commodity_manufactuer_df['match_supplier'] = input_commodity_manufactuer_df['supplier_name'].apply(
        lambda x: 'Supplier registered' if pd.notna(x) else 'No supplier found'
    )
    # Normalize the 'INV UOM' column to handle case sensitivity and strip spaces
    input_commodity_manufactuer_df['inv_uom'] = input_commodity_manufactuer_df['inv_uom'].str.strip(
    ).str.upper()
    # Classify rows based on 'INV UOM' values
    input_commodity_manufactuer_df['classification'] = input_commodity_manufactuer_df.apply(
        lambda row: 'Classified' if row['inv_uom'] in ['SQFT', 'SQYD']
        else ('No UOM' if pd.isna(row['inv_uom']) or row['inv_uom'] == '' else 'Unclassified'),
        axis=1
    )
    input_commodity_manufactuer_df['new_commodity_description'] = input_commodity_manufactuer_df.apply(
        classify_commodity, axis=1)
    input_commodity_manufactuer_df['new_commodity_group'] = input_commodity_manufactuer_df['commodity_group'].apply(
        map_commodity_group)

# Create a new column 'conversion_code' based on the 'Description' + 'Comodity Group' + 'INV UOM' column
    input_commodity_manufactuer_df['conversion_code'] = input_commodity_manufactuer_df['new_commodity_description'].str.replace(' ', '_', regex=True).astype(
        str) + '_' + input_commodity_manufactuer_df['new_commodity_group'].astype(str) + '_' + input_commodity_manufactuer_df['inv_uom'].astype(str)

    logging.info("✅ data_cleaning complete.")

    return input_commodity_manufactuer_df


def uom_cleaning(df):
    logging.info("✅ fixing unit of measure.")
    # checking which of the rows in an invoice matching 2008 has unclassified items
    # Check if all rows with account == 2008 are classified
    # Step 1: Identify invoice_ids where ALL rows with ACCOUNT == 2008 are classified
    uom_output = df
    classified_invoice_ids = (
        uom_output[uom_output['account'] == 2008]
        .groupby('invoice_id')['classification']
        .apply(lambda x: all(x == 'Classified'))
    )

    # Step 2: Filter to only invoice IDs where ALL 2008 accounts are classified
    fully_classified_ids = classified_invoice_ids[classified_invoice_ids].index

    # Step 3: Create a new column to mark if entire invoice is considered classified (based on the 2008 rule)
    uom_output['all_accounts_2008_uom_classified'] = uom_output['invoice_id'].isin(
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


def filter_valid_invoices(mapped_df):
    site_list = ['DIT', 'SPJ', 'SPN', 'SPT', 'SPW',
                 'SPCP', 'SPHU', 'KUS', 'PVF', 'SPTM']

    # Apply the filters
    filtered_df = mapped_df[
        (mapped_df['all_accounts_2008_uom_classified'] == True) &
        (mapped_df['all_2008_accounts_converted'] == True) &
        (mapped_df['all__invoice_priority_products_(2008)'] == True) &
        (mapped_df['has_freight_line'] == True) &
        (mapped_df['site'].isin(site_list)) &
        (mapped_df['invoiced_line_qty'] > 0) &
        (mapped_df['freight_per_invoice'] > 0)
    ]
    filtered_df = filtered_df[filtered_df['conversion_code'] != 'nan_nan_nan']

    return filtered_df
