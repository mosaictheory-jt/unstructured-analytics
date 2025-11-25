"""Convert CSV data to natural English sentences."""

import logging
from typing import Any

import pandas as pd

from .data_loader import load_all_tables, load_metadata

logger = logging.getLogger(__name__)


def _format_value(value: Any, field_type: str) -> str:
    """Format a value based on its type for natural language."""
    if pd.isna(value):
        return "not specified"
    
    if field_type == "decimal":
        return f"${value:,.2f}" if isinstance(value, (int, float)) else str(value)
    elif field_type == "integer":
        return f"{int(value):,}"
    elif field_type == "date":
        return str(value)
    else:
        return str(value)


def convert_customer_to_english(row: pd.Series) -> str:
    """Convert a customer record to an English sentence."""
    return (
        f"{row['first_name']} {row['last_name']} (ID: {row['customer_id']}) is a {row['customer_segment']} "
        f"customer who joined on {row['join_date']}. They are located in {row['city']}, {row['state']}, "
        f"{row['country']} and can be reached at {row['email']}."
    )


def convert_product_to_english(row: pd.Series) -> str:
    """Convert a product record to an English sentence."""
    margin = row['unit_price'] - row['cost_price']
    margin_pct = (margin / row['cost_price']) * 100
    return (
        f"The product '{row['product_name']}' (ID: {row['product_id']}) is a {row['subcategory']} item "
        f"in the {row['category']} category, manufactured by {row['brand']}. It is priced at "
        f"${row['unit_price']:.2f} with a cost of ${row['cost_price']:.2f}, yielding a margin of "
        f"${margin:.2f} ({margin_pct:.1f}%). This product is supplied by {row['supplier_id']}."
    )


def convert_supplier_to_english(row: pd.Series) -> str:
    """Convert a supplier record to an English sentence."""
    return (
        f"{row['supplier_name']} (ID: {row['supplier_id']}) is a supplier based in {row['country']}. "
        f"They have a lead time of {row['lead_time_days']} days and a reliability rating of "
        f"{row['reliability_rating']}/5.0. Contact them at {row['contact_email']}."
    )


def convert_order_to_english(row: pd.Series, customers_df: pd.DataFrame) -> str:
    """Convert an order record to an English sentence."""
    customer = customers_df[customers_df['customer_id'] == row['customer_id']].iloc[0]
    customer_name = f"{customer['first_name']} {customer['last_name']}"
    
    discount_text = f" with a discount of ${row['discount_applied']:.2f} applied" if row['discount_applied'] > 0 else ""
    
    return (
        f"Order {row['order_id']} was placed by {customer_name} ({row['customer_id']}) on {row['order_date']}. "
        f"The order uses {row['shipping_method']} shipping to {row['shipping_address_city']} and was paid via "
        f"{row['payment_method']}. Current status: {row['order_status']}{discount_text}."
    )


def convert_order_item_to_english(
    row: pd.Series, 
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame
) -> str:
    """Convert an order item record to an English sentence."""
    product = products_df[products_df['product_id'] == row['product_id']].iloc[0]
    order = orders_df[orders_df['order_id'] == row['order_id']].iloc[0]
    customer = customers_df[customers_df['customer_id'] == order['customer_id']].iloc[0]
    
    return (
        f"In order {row['order_id']}, {customer['first_name']} {customer['last_name']} purchased "
        f"{row['quantity']} unit(s) of '{product['product_name']}' at ${row['unit_price_at_sale']:.2f} each, "
        f"totaling ${row['total_price']:.2f}."
    )


def convert_all_to_english() -> str:
    """Convert all CSV data to natural English sentences."""
    tables = load_all_tables()
    
    customers_df = tables['customers']
    products_df = tables['products']
    suppliers_df = tables['suppliers']
    orders_df = tables['orders']
    order_items_df = tables['order_items']
    
    sections = []
    
    # Customers section
    customer_sentences = [convert_customer_to_english(row) for _, row in customers_df.iterrows()]
    sections.append("## Customers\n\n" + "\n\n".join(customer_sentences))
    
    # Suppliers section
    supplier_sentences = [convert_supplier_to_english(row) for _, row in suppliers_df.iterrows()]
    sections.append("## Suppliers\n\n" + "\n\n".join(supplier_sentences))
    
    # Products section
    product_sentences = [convert_product_to_english(row) for _, row in products_df.iterrows()]
    sections.append("## Products\n\n" + "\n\n".join(product_sentences))
    
    # Orders section
    order_sentences = [convert_order_to_english(row, customers_df) for _, row in orders_df.iterrows()]
    sections.append("## Orders\n\n" + "\n\n".join(order_sentences))
    
    # Order items section
    order_item_sentences = [
        convert_order_item_to_english(row, products_df, orders_df, customers_df) 
        for _, row in order_items_df.iterrows()
    ]
    sections.append("## Order Line Items\n\n" + "\n\n".join(order_item_sentences))
    
    return "\n\n---\n\n".join(sections)


def get_summary_statistics() -> str:
    """Generate a summary of the data in English."""
    tables = load_all_tables()
    
    order_items = tables['order_items']
    orders = tables['orders']
    products = tables['products']
    customers = tables['customers']
    
    total_revenue = order_items['total_price'].sum()
    total_orders = len(orders)
    avg_order_value = total_revenue / total_orders
    
    # Top selling products
    product_sales = order_items.groupby('product_id')['total_price'].sum().sort_values(ascending=False)
    top_product_id = product_sales.index[0]
    top_product = products[products['product_id'] == top_product_id]['product_name'].iloc[0]
    
    # Customer segment distribution
    segment_counts = customers['customer_segment'].value_counts()
    
    return f"""
## Data Summary

The e-commerce database contains:
- {len(customers)} customers across {len(customers['state'].unique())} states
- {len(products)} products from {len(tables['suppliers'])} suppliers
- {total_orders} orders with {len(order_items)} line items
- Total revenue: ${total_revenue:,.2f}
- Average order value: ${avg_order_value:,.2f}

Customer segments: {', '.join(f'{seg}: {count}' for seg, count in segment_counts.items())}

The best-selling product by revenue is "{top_product}".
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    english_output = convert_all_to_english()
    logger.info(f"Generated {len(english_output)} characters of English text")
    logger.info(english_output[:2000])

