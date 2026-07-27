"""Generate the DesiCart corpus: policy markdown in data/docs/ and data/orders.db.

Runs on a laptop, never in the container. Deterministic: same seed -> same DB.

Every fact in the generated docs comes from the CANONICAL FACTS block below, so the
corpus cannot contradict itself. Change a constant, re-run, and every doc that
mentions it updates together.

Usage:
    python scripts/make_corpus.py [--out-dir data]
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------------------
# CANONICAL FACTS - single source of truth for the whole corpus
# --------------------------------------------------------------------------------------

STORE = "DesiCart"
SUPPORT_EMAIL = "help@desicart.in"
SUPPORT_HOURS = "9 AM to 9 PM IST, all seven days"

# Returns / exchanges
RETURN_WINDOW_DAYS = 7
APPAREL_RETURN_WINDOW_DAYS = 15
EXCHANGE_WINDOW_DAYS = 7
DAMAGE_REPORT_HOURS = 48
PICKUP_ATTEMPTS = 3
PICKUP_SLA_DAYS = 4  # business days from return approval to first pickup attempt
QC_DAYS = 2  # business days for warehouse quality check after pickup
RETURN_PICKUP_FEE = 0  # free for all approved returns
NON_RETURNABLE = [
    "innerwear and lingerie",
    "groceries and perishables",
    "opened personal care and beauty products",
    "gift cards",
    "digital downloads",
    "made-to-order furniture",
]

# Refunds
REFUND_WALLET_HOURS = 2
REFUND_PREPAID_DAYS = "3 to 5"  # business days after QC pass
REFUND_COD_DAYS = "5 to 7"  # business days to the bank account you provide
REFUND_EMI_CYCLES = 2  # billing cycles for EMI reversal to appear
PAYMENT_REVERSAL_DAYS = "5 to 7"  # money debited but order not created

# Cancellations
CANCEL_FREE_UNTIL = "packed"  # free self-service cancellation up to and including this status
CANCEL_FEE = 0
PAYMENT_PENDING_AUTO_CANCEL_MINUTES = 60
PAYMENT_RETRY_WINDOW_MINUTES = 30

# Shipping
FREE_SHIPPING_THRESHOLD = 499
SHIPPING_FEE = 49
DELIVERY_ATTEMPTS = 3
ZONES: dict[str, tuple[str, int, int]] = {
    # key -> (label, min days, max days)
    "metro": ("Metro", 1, 2),
    "A": ("Zone A", 2, 4),
    "B": ("Zone B", 4, 6),
    "C": ("Zone C", 6, 9),
}
PEAK_EXTRA_DAYS = 2

# Cash on delivery
COD_MAX_ORDER_VALUE = 15000
COD_HANDLING_FEE = 29
COD_FEE_WAIVER_THRESHOLD = 499
COD_BLOCKED_CATEGORIES = ["jewellery", "gift cards", "mobiles priced above 15,000"]

# Warranty
WARRANTY_MONTHS = {
    "electronics": 12,
    "large appliances": 24,
    "small appliances": 6,
    "furniture": 12,
    "apparel": 0,
    "home": 0,
    "beauty": 0,
    "grocery": 0,
    "toys": 0,
}
WARRANTY_CLAIM_RESPONSE_DAYS = 3

# Gift cards
GIFT_CARD_MIN = 100
GIFT_CARD_MAX = 10000
GIFT_CARD_VALIDITY_MONTHS = 12
GIFT_CARDS_PER_ORDER = 5
GIFT_CARDS_PER_CHECKOUT = 3

# Seller
SELLER_REVIEW_DAYS = 7  # business days to review a new listing
SELLER_COMMISSION_RANGE = "5% to 18%"
SELLER_SETTLEMENT_DAYS = 7  # T+7 after delivery confirmation
SELLER_LATE_DISPATCH_PENALTY = "2% of order value"

# Sales
SALE_EVENTS = ["Diwali Dhamaka", "Republic Day Bazaar", "Monsoon Mela"]
PRICE_DROP_PROTECTION_DAYS = 7

# Account and privacy
ACCOUNT_DELETION_DAYS = 30
INVOICE_RETENTION_MONTHS = 96  # statutory
MARKETING_OPTOUT_DAYS = 7

# Serviceability. (state, zone key, cod allowed)
STATE_ZONES: list[tuple[str, str, bool]] = [
    ("Delhi", "metro", True),
    ("Maharashtra", "metro", True),
    ("Karnataka", "metro", True),
    ("Telangana", "metro", True),
    ("Tamil Nadu", "metro", True),
    ("West Bengal", "metro", True),
    ("Gujarat", "A", True),
    ("Rajasthan", "A", True),
    ("Uttar Pradesh", "A", True),
    ("Madhya Pradesh", "A", True),
    ("Punjab", "A", True),
    ("Haryana", "A", True),
    ("Kerala", "A", True),
    ("Andhra Pradesh", "A", True),
    ("Bihar", "B", True),
    ("Odisha", "B", True),
    ("Jharkhand", "B", True),
    ("Chhattisgarh", "B", True),
    ("Assam", "B", False),
    ("Uttarakhand", "B", True),
    ("Himachal Pradesh", "B", False),
    ("Goa", "B", True),
    ("Jammu and Kashmir", "C", False),
    ("Manipur", "C", False),
    ("Nagaland", "C", False),
    ("Sikkim", "C", False),
]
NON_SERVICEABLE_STATES = ["Andaman and Nicobar Islands", "Lakshadweep", "Ladakh"]

# Topics we deliberately do NOT document, so "the docs don't cover this" is a real case:
# international shipping, subscription plans, B2B / bulk orders, corporate gifting,
# price matching against other stores, and marketplace API access.

SEED = 4412
START_ORDER_ID = 4301
N_ORDERS = 200
DEMO_ORDER_ID = 4412  # README walkthrough uses this one; forced to a delivered order

# --------------------------------------------------------------------------------------
# Markdown docs
# --------------------------------------------------------------------------------------


def anchor(heading: str) -> str:
    """GitHub-style heading anchor, so citations can deep-link into a doc."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[\s_]+", "-", slug).strip("-")


def zone_line(key: str) -> str:
    label, lo, hi = ZONES[key]
    return f"{label}: {lo} to {hi} business days"


def returns_policy() -> str:
    return f"""# Returns Policy

This policy explains what you can return to {STORE}, how long you have, and what happens
after we collect the item.

## Return window

Most items can be returned within **{RETURN_WINDOW_DAYS} days** of delivery. Apparel and
footwear have a longer window of **{APPAREL_RETURN_WINDOW_DAYS} days** of delivery,
because sizing issues take longer to notice. The window is counted from the delivery date
recorded against your order, not from the date you opened the package. If the last day of
the window falls on a public holiday, you get until the next working day.

## What can be returned

An item is eligible for return if it is unused, in its original packaging, and has all
tags, manuals, accessories and freebies that shipped with it. Serial numbers on the box
must match the item inside. Items that were part of a combo must be returned as a complete
combo; we cannot accept a partial return of a bundled offer.

## What cannot be returned

The following are not returnable under any circumstances once delivered:
{chr(10).join(f"- {item}" for item in NON_RETURNABLE)}

If one of these arrives damaged or defective, it is handled under the damaged and
defective items policy instead, not as a return.

## How to raise a return request

Open your order in the {STORE} app or website, choose the item, and select Return. Pick a
reason and a pickup address. You will get a confirmation with a return ID. Returns raised
over email or phone are also accepted during support hours ({SUPPORT_HOURS}) at
{SUPPORT_EMAIL}, but the app is faster because you can attach photos directly.

## Pickup and quality check

Approved returns are picked up from your address at no cost to you. The pickup fee is
{RETURN_PICKUP_FEE} rupees. Our courier will make up to {PICKUP_ATTEMPTS} pickup attempts,
with the first attempt within {PICKUP_SLA_DAYS} business days of approval. If all attempts
fail, the return is closed and you will need to raise it again within the original return
window. After pickup the item goes through a quality check at the warehouse, which takes
{QC_DAYS} business days.

## Return charges

There are no return shipping charges on approved returns. If the quality check fails
because the item is used, damaged by the customer, or missing parts, the item is shipped
back to you and no refund is issued. You will be told the exact reason the check failed.
"""


def refunds() -> str:
    return f"""# Refunds and Refund Timelines

How and when {STORE} sends your money back.

## When a refund is issued

A refund is issued after one of the following: a cancelled order, a return that passed the
warehouse quality check, an undelivered shipment returned to origin, or a failed payment
where money was debited. Refund processing starts only after the quality check completes,
which takes {QC_DAYS} business days after pickup.

## Refund timelines by payment method

Timelines are counted from the moment the refund is initiated, not from the pickup date.

- {STORE} Wallet: credited within {REFUND_WALLET_HOURS} hours.
- UPI, credit card, debit card and netbanking: {REFUND_PREPAID_DAYS} business days.
- Cash on delivery: {REFUND_COD_DAYS} business days to the bank account you provide.
- EMI on credit card: the reversal appears within {REFUND_EMI_CYCLES} billing cycles. Your
  bank may still charge EMI interest already accrued.

## Refunds to DesiCart Wallet

You can choose to take any refund as {STORE} Wallet credit instead of a bank refund. Wallet
credit lands within {REFUND_WALLET_HOURS} hours and never expires. Wallet credit cannot be
withdrawn to a bank account once chosen, so pick this only if you plan to shop again.

## Partial refunds

If you return one item from a multi-item order, only that item's value is refunded.
Shipping charges of {SHIPPING_FEE} rupees are refunded only when every item in the order is
returned or cancelled. The cash on delivery handling fee of {COD_HANDLING_FEE} rupees is
not refundable, because it covers the collection that already happened.

## Refund not received

If the timeline above has passed, check your bank statement for a credit against the
reference number shown on the refund screen. Banks occasionally post refunds against the
original transaction date rather than the refund date. If it is still missing, write to
{SUPPORT_EMAIL} with the order ID and the reference number and we will raise a trace with
the payment provider.
"""


def cancellations() -> str:
    return f"""# Order Cancellations

## Cancelling before dispatch

You can cancel an order yourself, free of charge, at any time up to and including the
**{CANCEL_FREE_UNTIL}** status. Open the order and select Cancel. The cancellation fee is
{CANCEL_FEE} rupees. Cancellation is immediate and cannot be undone; to order again you
place a fresh order at the current price.

## Cancelling after dispatch

Once an order moves to in transit you can no longer cancel it from the app. You have two
options. You can refuse the package at the door, in which case it is returned to us and
refunded once it reaches the warehouse, or you can accept it and raise a return within the
normal return window of {RETURN_WINDOW_DAYS} days ({APPAREL_RETURN_WINDOW_DAYS} days for
apparel and footwear).

## Seller-initiated cancellations

A seller may cancel an order if the item turns out to be out of stock or the pincode is not
serviceable. You are notified by email and SMS, and a full refund including all shipping
and handling charges is initiated the same day. Seller cancellations count against the
seller's performance score.

## Cancellation refunds

Cancellations skip the quality check step because nothing was delivered, so refunds start
immediately. The timelines are the same as any other refund: {REFUND_WALLET_HOURS} hours
for wallet, {REFUND_PREPAID_DAYS} business days for prepaid methods. Cash on delivery
orders have nothing to refund unless the order was already paid for.

## Cancellation charges

There are no cancellation charges for buyer-initiated cancellations before dispatch. For
orders refused at the door after dispatch, the shipping charge of {SHIPPING_FEE} rupees is
not refunded if the order was below the free shipping threshold of
{FREE_SHIPPING_THRESHOLD} rupees.
"""


def shipping() -> str:
    state_rows = "\n".join(
        f"- {state}: {ZONES[zone][0]}, {ZONES[zone][1]} to {ZONES[zone][2]} business days, "
        f"cash on delivery {'available' if cod else 'not available'}."
        for state, zone, cod in STATE_ZONES
    )
    return f"""# Shipping Zones and Delivery Timelines

## Delivery zones

Every serviceable pincode belongs to one of four zones. The zone decides the delivery
estimate you see at checkout.

- {zone_line("metro")}. Covers Delhi, Mumbai, Bengaluru, Hyderabad, Chennai and Kolkata.
- {zone_line("A")}. State capitals and larger tier-two cities.
- {zone_line("B")}. Other district towns and their surrounding areas.
- {zone_line("C")}. Remote areas, hill districts and the north-east.

Business days exclude Sundays and national holidays. The clock starts when the order is
packed, not when it is placed.

## Serviceable states and union territories

We currently deliver to the following states and union territories. The zone and cash on
delivery availability for each are listed below. Individual pincodes inside a state can
differ from the state default, so always check the pincode on the product page before
ordering.

{state_rows}

We do not deliver to {", ".join(NON_SERVICEABLE_STATES)} at this time. Orders placed to a
non-serviceable pincode are cancelled by the seller and refunded in full. If your pincode
shows as not serviceable but a neighbouring one works, you can ship to the neighbouring
address; we do not hold parcels at a facility for self collection.

## Shipping charges

Orders of {FREE_SHIPPING_THRESHOLD} rupees and above ship free anywhere we deliver. Below
that, a flat shipping charge of {SHIPPING_FEE} rupees applies per order, not per item. The
charge is shown before you pay and appears as a separate line on the invoice.

## Delivery attempts

Our courier makes up to {DELIVERY_ATTEMPTS} delivery attempts on consecutive working days.
You get an SMS before each attempt. After the final failed attempt the parcel is returned
to the seller and, for prepaid orders, refunded in full once it reaches the warehouse.

## Tracking your shipment

Tracking becomes live once the parcel is scanned at the first hub, usually within a few
hours of dispatch. Until then the order shows as packed with no tracking events. A parcel
that shows no scan for more than 72 hours is treated as a lost shipment and refunded.

## Delays during peak periods

During sale events and major festivals, delivery estimates are extended by about
{PEAK_EXTRA_DAYS} business days across all zones. The estimate shown at checkout already
includes this extension, so the date on your order page is the one to rely on.
"""


def cod() -> str:
    return f"""# Cash on Delivery

Cash on delivery lets you pay the courier when the parcel arrives, in cash or by UPI at the
door.

## COD eligibility

Cash on delivery is available on most orders shipping to a pincode that supports it. The
product page shows whether the option is available for your pincode before you add the item
to the cart. Cash on delivery is not offered in
{", ".join(s for s, _, c in STATE_ZONES if not c)}, or in any
{ZONES["C"][0]} pincode where the courier does not carry cash.

## COD order limits

The maximum order value payable by cash on delivery is **{COD_MAX_ORDER_VALUE:,} rupees**.
Orders above this must be prepaid. The limit applies to the order total after discounts,
including shipping and the handling fee. Splitting a large order into several smaller cash
on delivery orders to the same address may trigger a fraud check.

## COD handling fee

A handling fee of {COD_HANDLING_FEE} rupees applies to cash on delivery orders below
{COD_FEE_WAIVER_THRESHOLD} rupees. At or above {COD_FEE_WAIVER_THRESHOLD} rupees the fee is
waived. The fee covers cash collection and is not refunded if you later return the item.

## Items not eligible for COD

Some categories are always prepaid, regardless of pincode:
{chr(10).join(f"- {item}" for item in COD_BLOCKED_CATEGORIES)}

If your cart contains one of these, the cash on delivery option is hidden at checkout for
the whole cart.

## Refunds on COD orders

Because we never held your money, a cash on delivery refund is paid to a bank account you
provide during the return flow. It takes {REFUND_COD_DAYS} business days from refund
initiation. You can instead take the refund as wallet credit, which lands within
{REFUND_WALLET_HOURS} hours.

## Refused COD deliveries

You may refuse a cash on delivery parcel at the door without paying. Repeated refusals
across orders can result in cash on delivery being disabled for your account, after which
only prepaid checkout is available. We tell you by email before that happens.
"""


def damaged() -> str:
    return f"""# Damaged, Defective or Missing Items

## Reporting window

Report a damaged, defective or missing item within **{DAMAGE_REPORT_HOURS} hours** of
delivery. This is shorter than the normal return window of {RETURN_WINDOW_DAYS} days
because we need the courier's handover record while it is still available. Reports made
after {DAMAGE_REPORT_HOURS} hours are handled as a normal return, which means the item must
still be unused and in its original packaging to qualify.

## What we need from you

Attach photos of the outer packaging including the shipping label, the item itself showing
the damage, and the invoice. For a defective electronic item, add a short video showing the
fault. Keep the packaging until the case is closed; the courier may need to collect it with
the item.

## Resolution options

Once the report is accepted you choose one of three outcomes: a replacement of the same
item, a refund, or a repair under warranty if the item is covered. Replacements ship as
soon as the damaged item is picked up. Refunds follow the standard timelines,
{REFUND_PREPAID_DAYS} business days for prepaid orders.

## Missing items in a multi-item order

If part of a multi-item order is missing, check whether it shipped separately. Orders often
split across shipments and each has its own tracking. If the packing slip lists an item that
was not in the box, report it within {DAMAGE_REPORT_HOURS} hours and we will either ship the
missing item or refund that line, at your choice.

## Open box deliveries

For high-value electronics, the courier opens the box in front of you and you check the item
before paying or signing. If the item is damaged at that point, refuse the delivery. A
refused open box delivery is refunded in full without a return pickup.
"""


def exchanges() -> str:
    return f"""# Exchanges

## Exchange window

Exchanges can be raised within **{EXCHANGE_WINDOW_DAYS} days** of delivery. For apparel and
footwear you may also use the longer {APPAREL_RETURN_WINDOW_DAYS} day window if you are
returning for a refund instead of exchanging.

## What can be exchanged

Exchanges are offered for size, colour or variant changes on apparel, footwear and
accessories, and for the same model on a defective electronic item. Anything in the
non-returnable list cannot be exchanged. An exchange is only possible if the variant you
want is in stock with the same seller.

## How an exchange works

The courier brings the replacement and collects the original item in the same visit, so you
are never without the product. If the replacement fails our checks at the door, the visit is
converted into a return and refunded. Exchange pickups follow the same
{PICKUP_ATTEMPTS} attempt rule as returns.

## Price differences on exchange

An exchange for a different variant at the same price has no payment step. If the new
variant costs more, you pay the difference at the door. If it costs less, the difference is
refunded to your original payment method within {REFUND_PREPAID_DAYS} business days.

## Exchange limits

One exchange is allowed per item. If the replacement also has a problem, the second request
is processed as a return and refund rather than another exchange.
"""


def warranty() -> str:
    rows = "\n".join(
        f"- {cat.capitalize()}: {months} months manufacturer warranty."
        if months
        else f"- {cat.capitalize()}: no warranty; covered only by the return policy."
        for cat, months in WARRANTY_MONTHS.items()
    )
    return f"""# Warranty and Repairs

## Warranty periods

Warranty is provided by the manufacturer, not by {STORE}. Standard periods by category:

{rows}

The warranty starts on the delivery date recorded against your order. Your {STORE} invoice
is valid proof of purchase at any authorised service centre.

## What warranty covers

Manufacturing defects, component failure under normal use, and functional faults present
from the start. If the fault appears within {RETURN_WINDOW_DAYS} days of delivery you can
choose a replacement or refund instead of a repair.

## What warranty does not cover

Physical damage, liquid damage, damage from a power surge, normal wear, consumables such as
batteries and filters, cosmetic marks, and any product opened or repaired by an
unauthorised technician. Loss or theft is never covered.

## Raising a warranty claim

Raise the claim from the order page or contact the manufacturer's service centre directly
with the {STORE} invoice. We respond to claims raised through {STORE} within
{WARRANTY_CLAIM_RESPONSE_DAYS} business days with a service centre reference or a pickup
slot. Turnaround at the service centre depends on the manufacturer.

## Replacement under warranty

If the service centre declares the item beyond economical repair, the manufacturer issues a
replacement of the same or an equivalent model. If no equivalent exists, {STORE} refunds the
invoice value to your original payment method within {REFUND_PREPAID_DAYS} business days of
receiving the service centre's report.
"""


def seller_onboarding() -> str:
    return f"""# Seller Onboarding

## Who can sell on DesiCart

Any registered business with a valid GSTIN can sell on {STORE}. Individuals selling
handmade goods can register under the composition scheme if their category is exempt.
Resellers of counterfeit or restricted goods are removed permanently and their settlements
are withheld.

## Documents required

You will need a GSTIN certificate, a PAN card in the business name, a cancelled cheque or
bank statement for the settlement account, and one address proof for the pickup location.
Brand owners must also upload a trademark certificate to unlock brand gating.

## Listing review

New listings are reviewed within {SELLER_REVIEW_DAYS} business days. Review checks images,
title format, category placement and mandatory attributes. A rejected listing comes back
with the exact field to fix and can be resubmitted immediately; resubmissions are usually
reviewed faster than first submissions.

## Commission and fees

Commission ranges from {SELLER_COMMISSION_RANGE} of the item price depending on category,
plus a fixed closing fee and shipping charges based on weight slab. The full fee schedule is
visible in the seller dashboard before you list. Fees are deducted at settlement, never
invoiced separately.

## Settlement cycle

Settlements run on a T plus {SELLER_SETTLEMENT_DAYS} cycle: the payout for an order is
released {SELLER_SETTLEMENT_DAYS} days after delivery is confirmed, once the return window
has been accounted for. Payouts land in the registered bank account and a settlement
statement is available in the dashboard.

## Seller performance standards

Sellers must dispatch within the promised handling time. Late dispatch carries a penalty of
{SELLER_LATE_DISPATCH_PENALTY} and repeated breaches reduce search visibility. A cancellation
rate above 2 percent or a return rate driven by wrong or damaged shipments triggers an
account review.
"""


def payment_failures() -> str:
    return f"""# Payment Failures and Debits

## Money debited but no order

If your account was debited but no order appears, the payment did not reach us and the bank
will reverse it automatically within {PAYMENT_REVERSAL_DAYS} business days. No action is
needed from you. Do not place the order again until you have checked your orders list, or
you may end up with two orders.

## Failed payment retries

You can retry a failed payment on the same cart for {PAYMENT_RETRY_WINDOW_MINUTES} minutes
using a different method without losing your prices or applied coupons. After that the cart
is repriced at current rates and the coupon may no longer be valid.

## Pending payment orders

An order stuck in pending payment is cancelled automatically after
{PAYMENT_PENDING_AUTO_CANCEL_MINUTES} minutes and any amount captured is refunded on the
standard timeline of {REFUND_PREPAID_DAYS} business days. You will see the cancellation in
your orders list with the reason shown as payment not confirmed.

## UPI and netbanking specifics

UPI failures are usually a timeout at the bank end and reverse fastest, often the same day.
Netbanking failures can take the full {PAYMENT_REVERSAL_DAYS} business days because the
reversal follows the bank's settlement file. Always keep the UPI reference number or bank
transaction ID; we cannot trace a payment without it.

## EMI and card issues

If an EMI order is cancelled or returned, the reversal appears within {REFUND_EMI_CYCLES}
billing cycles. Interest already charged by your bank for elapsed cycles is not refunded by
{STORE}; take that up with the card issuer. Cards that fail repeatedly are often blocked for
online use by the issuer rather than declined by us.
"""


def festival_sale() -> str:
    return f"""# Festival Sale Terms

## Sale events

{STORE} runs three flagship sale events each year: {", ".join(SALE_EVENTS)}. Sale prices are
live only for the announced window and revert automatically when it closes. Prices during a
sale are not honoured for orders placed before or after it.

## Doorbuster deals

Doorbuster deals are limited-quantity offers released at fixed times. They are limited to
one unit per account, cannot be combined with coupons, and are exchange-only if there is a
problem: a doorbuster item can be exchanged for the same model but is not eligible for a
refund unless no replacement stock exists.

## Price drop protection

If the price of an item you bought drops further within {PRICE_DROP_PROTECTION_DAYS} days of
your order during the same sale event, claim the difference from the order page and we
credit it to your {STORE} Wallet within {REFUND_WALLET_HOURS} hours. Price drop protection
does not apply to doorbuster deals or to price changes after the sale event ends.

## Coupons and bank offers

One coupon per order. Bank offers stack with coupons but are capped per card per event, and
the cap is shown on the offer banner. If an order that used a bank offer is returned, the
bank discount is deducted from the refund because the bank reverses its contribution.

## Returns during sale periods

The standard return window of {RETURN_WINDOW_DAYS} days ({APPAREL_RETURN_WINDOW_DAYS} days
for apparel and footwear) applies unchanged during sale events, except for doorbuster deals
as described above. Sale pricing does not reduce your return rights.

## Delivery timelines during sales

Expect about {PEAK_EXTRA_DAYS} extra business days across all zones during a sale event.
The estimate shown at checkout already includes the extension. Cash on delivery may be
temporarily unavailable on high-demand items to reduce refusal rates.
"""


def account_privacy() -> str:
    return f"""# Account and Privacy

## Creating and accessing your account

An account is created with a mobile number verified by OTP. Email is optional but is needed
for invoices and refund notifications. We never ask for your OTP, card CVV or UPI PIN;
anyone who does is not from {STORE}.

## Updating your details

You can change your name, email and addresses from the profile screen at any time. Changing
the registered mobile number requires OTP verification on both the old and the new number.
If you no longer have access to the old number, write to {SUPPORT_EMAIL} from the registered
email address with a photo ID.

## Deleting your account

Request deletion from the profile screen. We process deletion within
{ACCOUNT_DELETION_DAYS} days. Open orders, active returns and pending refunds must be closed
first, because we cannot refund to a deleted account. Deletion is permanent and wallet
balance is forfeited, so withdraw or spend it first.

## What data we keep

After deletion we retain invoices and transaction records for {INVOICE_RETENTION_MONTHS}
months, because tax law requires it. Everything else, including browsing history,
recommendations, saved addresses and support chats, is removed. Retained records are not
used for marketing or recommendations.

## Marketing preferences

Opt out of promotional email, SMS and push from notification settings. Changes take effect
within {MARKETING_OPTOUT_DAYS} days. Transactional messages about orders, deliveries and
refunds continue regardless, because they are needed to fulfil your order.

## Reporting a security concern

Report a suspected account compromise or a security issue to {SUPPORT_EMAIL} with the
subject line SECURITY. We acknowledge within one business day. If you believe your account
was accessed by someone else, change the registered number immediately and check your saved
addresses for entries you do not recognise.
"""


def gift_cards() -> str:
    return f"""# Gift Cards

## Buying a gift card

Gift cards are bought from the gift cards page and delivered by email and SMS to the
recipient, usually within a few minutes. Gift cards must be prepaid; cash on delivery is not
available for them, and they cannot be bought using another gift card.

## Denominations and limits

Gift cards are available from {GIFT_CARD_MIN} rupees to {GIFT_CARD_MAX:,} rupees. You can
buy up to {GIFT_CARDS_PER_ORDER} gift cards in a single order.

## Redeeming a gift card

Enter the card number and PIN at checkout. Up to {GIFT_CARDS_PER_CHECKOUT} gift cards can be
applied to one order, and any remaining amount is paid by another method. Balance left on a
card stays on the card for its remaining validity. A gift card cannot be used to buy another
gift card.

## Validity and expiry

Gift cards are valid for {GIFT_CARD_VALIDITY_MONTHS} months from the date of issue. The
expiry date is printed on the card and shown in your account. Balance left on an expired
card cannot be reinstated or extended.

## Refunds and cancellations

Gift cards are non-returnable and non-refundable, and the value cannot be transferred back
to a bank account or to wallet cash. If an order paid for with a gift card is returned, the
gift card portion of the refund goes back to the same gift card and keeps its original
expiry date.

## Lost or unused gift cards

If a gift card email was sent to the wrong address, write to {SUPPORT_EMAIL} within
{DAMAGE_REPORT_HOURS} hours of purchase with the order ID and we will cancel and reissue it,
provided it has not been redeemed. A redeemed card cannot be recovered.
"""


DOCS: dict[str, str] = {
    "returns-policy.md": returns_policy(),
    "refunds-and-timelines.md": refunds(),
    "order-cancellations.md": cancellations(),
    "shipping-zones-and-timelines.md": shipping(),
    "cash-on-delivery.md": cod(),
    "damaged-or-missing-items.md": damaged(),
    "exchanges.md": exchanges(),
    "warranty-and-repairs.md": warranty(),
    "seller-onboarding.md": seller_onboarding(),
    "payment-failures.md": payment_failures(),
    "festival-sale-terms.md": festival_sale(),
    "account-and-privacy.md": account_privacy(),
    "gift-cards.md": gift_cards(),
}

# --------------------------------------------------------------------------------------
# Catalogue and pincodes
# --------------------------------------------------------------------------------------

# (name, category, price)
CATALOGUE: list[tuple[str, str, int]] = [
    ("Redmi Note 14 5G 128GB", "electronics", 16999),
    ("boAt Airdopes 141 Earbuds", "electronics", 1299),
    ("Noise ColorFit Pro 5 Smartwatch", "electronics", 3499),
    ("HP 15s Ryzen 5 Laptop", "electronics", 44990),
    ("Logitech M235 Wireless Mouse", "electronics", 749),
    ("Anker 20000mAh Power Bank", "electronics", 2199),
    ("Samsung 32 inch HD Smart TV", "electronics", 13499),
    ("Sony WH-CH520 Headphones", "electronics", 4290),
    ("LG 7kg Front Load Washing Machine", "large appliances", 28990),
    ("Voltas 1.5 Ton Split AC", "large appliances", 34999),
    ("Godrej 190L Single Door Fridge", "large appliances", 16490),
    ("Prestige Induction Cooktop 1900W", "small appliances", 2499),
    ("Bajaj Mixer Grinder 750W", "small appliances", 3199),
    ("Philips Air Fryer HD9200", "small appliances", 8499),
    ("Havells Ceiling Fan 1200mm", "small appliances", 2099),
    ("Milton Thermosteel Flask 1L", "home", 899),
    ("Cotton Bedsheet Double Jaipuri Print", "home", 1099),
    ("Solimo Mattress 6 inch Queen", "home", 8999),
    ("Cello Storage Container Set of 6", "home", 649),
    ("Wakefit Study Table", "furniture", 5499),
    ("Nilkamal Plastic Chair Set of 2", "furniture", 1899),
    ("Levis Mens Slim Fit Jeans", "apparel", 2299),
    ("Biba Cotton Anarkali Kurta", "apparel", 1799),
    ("Puma Mens Running Shoes", "apparel", 2999),
    ("Allen Solly Womens Formal Shirt", "apparel", 1499),
    ("Jockey Mens Cotton Vest Pack of 3", "apparel", 699),
    ("Fastrack Analog Watch Black Dial", "apparel", 1795),
    ("Raymond Wool Blend Blazer", "apparel", 5499),
    ("Mamaearth Onion Hair Oil 250ml", "beauty", 399),
    ("Lakme Absolute Matte Lipstick", "beauty", 649),
    ("Nivea Soft Moisturiser 200ml", "beauty", 299),
    ("Dove Intense Repair Shampoo 650ml", "beauty", 749),
    ("Tata Sampann Toor Dal 5kg", "grocery", 899),
    ("Aashirvaad Atta 10kg", "grocery", 519),
    ("Tata Tea Gold 1kg", "grocery", 585),
    ("Saffola Gold Oil 5L", "grocery", 899),
    ("Funskool Rubiks Cube 3x3", "toys", 449),
    ("Hot Wheels Die Cast Pack of 5", "toys", 699),
    ("Lego Classic Creative Bricks", "toys", 2599),
    ("Camlin Kokuyo Art Set", "toys", 549),
]

SELLERS = [
    "Shreeji Retail LLP",
    "Nexus Traders Pvt Ltd",
    "Kaveri Electronics",
    "Bharat Home Mart",
    "Vastra Fashions",
    "Annapurna Foods",
]

# (pincode, city, state) - the zone and cod flag come from STATE_ZONES.
PINCODES: list[tuple[str, str, str]] = [
    ("110001", "New Delhi", "Delhi"),
    ("110024", "New Delhi", "Delhi"),
    ("110085", "New Delhi", "Delhi"),
    ("400001", "Mumbai", "Maharashtra"),
    ("400050", "Mumbai", "Maharashtra"),
    ("411001", "Pune", "Maharashtra"),
    ("440010", "Nagpur", "Maharashtra"),
    ("560001", "Bengaluru", "Karnataka"),
    ("560076", "Bengaluru", "Karnataka"),
    ("575001", "Mangaluru", "Karnataka"),
    ("500001", "Hyderabad", "Telangana"),
    ("500081", "Hyderabad", "Telangana"),
    ("600001", "Chennai", "Tamil Nadu"),
    ("600096", "Chennai", "Tamil Nadu"),
    ("641001", "Coimbatore", "Tamil Nadu"),
    ("700001", "Kolkata", "West Bengal"),
    ("700091", "Kolkata", "West Bengal"),
    ("380001", "Ahmedabad", "Gujarat"),
    ("395003", "Surat", "Gujarat"),
    ("302001", "Jaipur", "Rajasthan"),
    ("313001", "Udaipur", "Rajasthan"),
    ("226001", "Lucknow", "Uttar Pradesh"),
    ("208001", "Kanpur", "Uttar Pradesh"),
    ("221005", "Varanasi", "Uttar Pradesh"),
    ("452001", "Indore", "Madhya Pradesh"),
    ("462001", "Bhopal", "Madhya Pradesh"),
    ("160017", "Chandigarh", "Punjab"),
    ("141001", "Ludhiana", "Punjab"),
    ("122001", "Gurugram", "Haryana"),
    ("121001", "Faridabad", "Haryana"),
    ("682001", "Kochi", "Kerala"),
    ("695001", "Thiruvananthapuram", "Kerala"),
    ("530001", "Visakhapatnam", "Andhra Pradesh"),
    ("520001", "Vijayawada", "Andhra Pradesh"),
    ("800001", "Patna", "Bihar"),
    ("842001", "Muzaffarpur", "Bihar"),
    ("751001", "Bhubaneswar", "Odisha"),
    ("753001", "Cuttack", "Odisha"),
    ("834001", "Ranchi", "Jharkhand"),
    ("492001", "Raipur", "Chhattisgarh"),
    ("781001", "Guwahati", "Assam"),
    ("248001", "Dehradun", "Uttarakhand"),
    ("171001", "Shimla", "Himachal Pradesh"),
    ("403001", "Panaji", "Goa"),
    ("180001", "Jammu", "Jammu and Kashmir"),
    ("190001", "Srinagar", "Jammu and Kashmir"),
    ("795001", "Imphal", "Manipur"),
    ("797001", "Kohima", "Nagaland"),
    ("737101", "Gangtok", "Sikkim"),
    # Not serviceable at all.
    ("744101", "Port Blair", "Andaman and Nicobar Islands"),
    ("682555", "Kavaratti", "Lakshadweep"),
    ("194101", "Leh", "Ladakh"),
]

STATUSES = [
    ("delivered", 0.40),
    ("in_transit", 0.12),
    ("placed", 0.06),
    ("packed", 0.07),
    ("cancelled", 0.09),
    ("returned", 0.11),
    ("refunded", 0.15),
]

PAYMENT_METHODS = ["upi", "credit_card", "debit_card", "netbanking", "wallet", "cod"]

SCHEMA = """
CREATE TABLE products (
    product_id      INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    price_inr       INTEGER NOT NULL,
    seller          TEXT    NOT NULL,
    warranty_months INTEGER NOT NULL,
    returnable      INTEGER NOT NULL
);

CREATE TABLE orders (
    order_id          INTEGER PRIMARY KEY,
    customer_name     TEXT    NOT NULL,
    status            TEXT    NOT NULL,
    payment_method    TEXT    NOT NULL,
    order_value_inr   INTEGER NOT NULL,
    shipping_fee_inr  INTEGER NOT NULL,
    cod_fee_inr       INTEGER NOT NULL,
    pincode           TEXT    NOT NULL,
    city              TEXT    NOT NULL,
    state             TEXT    NOT NULL,
    zone              TEXT    NOT NULL,
    placed_at         TEXT    NOT NULL,
    packed_at         TEXT,
    shipped_at        TEXT,
    expected_delivery TEXT    NOT NULL,
    delivered_at      TEXT,
    cancelled_at      TEXT,
    returned_at       TEXT,
    refunded_at       TEXT,
    refund_amount_inr INTEGER
);

CREATE TABLE order_items (
    item_id        INTEGER PRIMARY KEY,
    order_id       INTEGER NOT NULL REFERENCES orders(order_id),
    product_id     INTEGER NOT NULL REFERENCES products(product_id),
    quantity       INTEGER NOT NULL,
    unit_price_inr INTEGER NOT NULL,
    line_total_inr INTEGER NOT NULL
);

CREATE TABLE pincodes (
    pincode      TEXT PRIMARY KEY,
    city         TEXT    NOT NULL,
    state        TEXT    NOT NULL,
    zone         TEXT    NOT NULL,
    serviceable  INTEGER NOT NULL,
    cod_available INTEGER NOT NULL,
    est_days_min INTEGER NOT NULL,
    est_days_max INTEGER NOT NULL
);

CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_placed_at ON orders(placed_at);
CREATE INDEX idx_order_items_order ON order_items(order_id);
"""

FIRST_NAMES = [
    "Aarav", "Vihaan", "Ananya", "Diya", "Ishaan", "Kavya", "Rohan", "Meera",
    "Arjun", "Sneha", "Karthik", "Priya", "Rahul", "Neha", "Aditya", "Pooja",
    "Farhan", "Zoya", "Manish", "Ritu", "Sandeep", "Divya", "Nikhil", "Anjali",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Nair", "Reddy", "Patel", "Gupta", "Bose",
    "Khan", "Mehta", "Joshi", "Das", "Rao", "Singh", "Chopra", "Pillai",
]


def weighted_status(rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for status, weight in STATUSES:
        cumulative += weight
        if r <= cumulative:
            return status
    return STATUSES[-1][0]


def build_db(db_path: Path, rng: random.Random) -> dict[str, int]:
    if db_path.exists():
        db_path.unlink()  # regenerate from scratch; the DB is a build artefact, not state
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # Products
    products: list[tuple[int, str, str, int, str, int, int]] = []
    for i, (name, category, price) in enumerate(CATALOGUE, start=1):
        returnable = 0 if category in ("grocery", "beauty") and price < 500 else 1
        products.append(
            (i, name, category, price, SELLERS[i % len(SELLERS)], WARRANTY_MONTHS[category], returnable)
        )
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", products)

    # Pincodes
    state_zone = {state: (zone, cod) for state, zone, cod in STATE_ZONES}
    pin_rows = []
    for pincode, city, state in PINCODES:
        if state in NON_SERVICEABLE_STATES:
            pin_rows.append((pincode, city, state, "none", 0, 0, 0, 0))
            continue
        zone, cod_allowed = state_zone[state]
        lo, hi = ZONES[zone][1], ZONES[zone][2]
        pin_rows.append((pincode, city, state, zone, 1, int(cod_allowed), lo, hi))
    conn.executemany("INSERT INTO pincodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pin_rows)

    serviceable_pins = [r for r in pin_rows if r[4] == 1]

    today = date.today()
    orders: list[tuple] = []
    items: list[tuple] = []
    item_id = 1

    for n in range(N_ORDERS):
        order_id = START_ORDER_ID + n
        status = "delivered" if order_id == DEMO_ORDER_ID else weighted_status(rng)
        pincode, city, state, zone, _, cod_ok, lo, hi = rng.choice(serviceable_pins)

        # Lines
        n_lines = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        chosen = rng.sample(range(len(products)), n_lines)
        subtotal = 0
        order_lines = []
        for idx in chosen:
            pid, _, _, price, _, _, _ = products[idx]
            qty = rng.choices([1, 2], weights=[0.85, 0.15])[0]
            line_total = price * qty
            subtotal += line_total
            order_lines.append((item_id, order_id, pid, qty, price, line_total))
            item_id += 1

        shipping_fee = 0 if subtotal >= FREE_SHIPPING_THRESHOLD else SHIPPING_FEE

        cod_eligible = bool(cod_ok) and subtotal + shipping_fee <= COD_MAX_ORDER_VALUE
        methods = PAYMENT_METHODS if cod_eligible else [m for m in PAYMENT_METHODS if m != "cod"]
        payment_method = rng.choice(methods)
        cod_fee = (
            COD_HANDLING_FEE
            if payment_method == "cod" and subtotal < COD_FEE_WAIVER_THRESHOLD
            else 0
        )
        order_value = subtotal + shipping_fee + cod_fee

        # Dates. Every gap in the chain is drawn first, so the order can be placed far
        # enough back that its last event has already happened - no future timestamps.
        transit_days = rng.randint(lo, hi)
        slip = rng.choice([-1, 0, 0, 0, 1, 2])  # most on time, a few late
        actual_transit = max(1, transit_days + slip)
        return_gap = rng.randint(1, RETURN_WINDOW_DAYS)
        refund_gap = rng.randint(2, 7)

        min_age = 0
        if status in ("delivered", "returned", "refunded"):
            min_age = actual_transit + 2
        if status in ("returned", "refunded"):
            min_age += return_gap
        if status == "refunded":
            min_age += refund_gap
        max_age = {"placed": 2, "packed": 3, "in_transit": 7, "cancelled": 60}.get(status, 90)
        if order_id == DEMO_ORDER_ID:
            # Keep the walkthrough order inside the return window so the demo query is live.
            max_age = min_age + 1
        max_age = max(max_age, min_age)
        age_days = rng.randint(min_age, max_age)
        placed = today - timedelta(days=age_days)
        placed_at = datetime.combine(placed, datetime.min.time()) + timedelta(
            hours=rng.randint(8, 21), minutes=rng.randint(0, 59)
        )
        expected_delivery = placed + timedelta(days=transit_days + 1)

        packed_at = shipped_at = delivered_at = None
        cancelled_at = returned_at = refunded_at = None
        refund_amount = None

        if status in ("packed", "in_transit", "delivered", "returned", "refunded"):
            packed_at = placed_at + timedelta(hours=rng.randint(6, 30))
        if status in ("in_transit", "delivered", "returned", "refunded"):
            shipped_at = packed_at + timedelta(hours=rng.randint(4, 20))
        # Events at least a day apart get a plausible daytime clock; same-day offsets keep
        # their computed hour so the ordering can never invert.
        def daytime(value: datetime) -> datetime:
            return value.replace(hour=rng.randint(9, 20), minute=rng.randint(0, 59), second=0)

        if status in ("delivered", "returned", "refunded"):
            delivered_at = daytime(shipped_at + timedelta(days=actual_transit))
        if status in ("returned", "refunded") and delivered_at is not None:
            returned_at = daytime(delivered_at + timedelta(days=return_gap))
        if status == "cancelled":
            cancelled_at = placed_at + timedelta(hours=rng.randint(1, 48))
            packed_at = None
        if status == "refunded":
            base = returned_at or cancelled_at or placed_at
            refunded_at = daytime(base + timedelta(days=refund_gap))
            refund_amount = order_value - cod_fee  # COD handling fee is never refunded

        def iso(value: datetime | None) -> str | None:
            return value.isoformat(sep=" ", timespec="seconds") if value else None

        orders.append(
            (
                order_id,
                f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                status,
                payment_method,
                order_value,
                shipping_fee,
                cod_fee,
                pincode,
                city,
                state,
                zone,
                iso(placed_at),
                iso(packed_at),
                iso(shipped_at),
                expected_delivery.isoformat(),
                iso(delivered_at),
                iso(cancelled_at),
                iso(returned_at),
                iso(refunded_at),
                refund_amount,
            )
        )
        items.extend(order_lines)

    conn.executemany(
        "INSERT INTO orders VALUES (" + ",".join("?" * 20) + ")",
        orders,
    )
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", items)
    conn.commit()

    counts = {
        "products": len(products),
        "orders": len(orders),
        "order_items": len(items),
        "pincodes": len(pin_rows),
    }
    conn.close()
    return counts


def write_docs(docs_dir: Path) -> list[tuple[str, int, int]]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for stale in docs_dir.glob("*.md"):
        stale.unlink()
    summary = []
    for filename, body in DOCS.items():
        path = docs_dir / filename
        path.write_text(body, encoding="utf-8")
        sections = body.count("\n## ")
        summary.append((filename, sections, len(body.split())))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DesiCart corpus.")
    parser.add_argument("--out-dir", default="data", help="output directory (default: data)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    rng = random.Random(SEED)

    doc_summary = write_docs(out_dir / "docs")
    counts = build_db(out_dir / "orders.db", rng)

    print(f"docs -> {out_dir / 'docs'}")
    for filename, sections, words in doc_summary:
        print(f"  {filename:<34} {sections:>2} sections  {words:>5} words")
    print(f"\ndb   -> {out_dir / 'orders.db'}")
    for table, n in counts.items():
        print(f"  {table:<12} {n:>5} rows")

    conn = sqlite3.connect(out_dir / "orders.db")
    print("\n  status breakdown:")
    for status, n in conn.execute(
        "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY COUNT(*) DESC"
    ):
        print(f"    {status:<12} {n:>4}")
    demo = conn.execute(
        "SELECT status, delivered_at FROM orders WHERE order_id = ?", (DEMO_ORDER_ID,)
    ).fetchone()
    print(f"\n  demo order {DEMO_ORDER_ID}: status={demo[0]} delivered_at={demo[1]}")
    conn.close()


if __name__ == "__main__":
    main()
