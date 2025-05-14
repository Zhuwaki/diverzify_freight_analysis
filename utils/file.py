
  for _, row in invoice_df.iterrows():
       try:
            invoice_id = row['invoice_id']
            site = row['site'].upper()
            group = row['new_commodity_group'].upper()
            qty = row['invoice_commodity_quantity']
            method = row['method']
            unit = row['unit']
            rate_unit = row['rate_unit']

            if method == 'UNKNOWN' or unit == 'UNKNOWN' or qty is None or qty == 0:
                enriched_rows.append({**row,
                                      'freight_class': None,
                                      'rate': None,
                                      'shipment_type': None,
                                      'invoice_freight_commodity_cost': None})
                continue

            freight_class = get_freight_class(qty)
            rate, error = get_freight_rate(
                site, rate_unit, group, freight_class)

            if error:
                enriched_rows.append({**row,
                                      'freight_class': None,
                                      'rate': None,
                                      'shipment_type': None,
                                      'raw_invoice_cost': None,
                                      'invoice_freight_commodity_cost': None,
                                      'minimum_applied': None})
                continue

            if method == 'CWT':
                rate = rate / 100  # Adjust for per 100lbs

            shipment_type = classify_shipment_by_uom(qty, unit)

            raw_invoice_cost = round(rate * qty, 2)
            min_charge = minimum_charges.get(site, {}).get(group, 0)
            if raw_invoice_cost < min_charge:
                invoice_freight_commodity_cost = min_charge
                minimum_applied = True
            else:
                invoice_freight_commodity_cost = raw_invoice_cost
                minimum_applied = False

            enriched_rows.append({
                **row,
                'freight_class': freight_class,
                'rate': rate,
                'shipment_type': shipment_type,
                'raw_invoice_cost': raw_invoice_cost,
                'invoice_freight_commodity_cost': invoice_freight_commodity_cost,
                'minimum_applied': minimum_applied
            })

        except Exception:
            enriched_rows.append({**row,
                                  'freight_class': None,
                                  'rate': None,
                                  'shipment_type': None,
                                  'invoice_freight_commodity_cost': None})
