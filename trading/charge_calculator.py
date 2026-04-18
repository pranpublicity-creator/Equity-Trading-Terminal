"""
NSE Equity Charge Calculator
Calculates all applicable charges for equity trades on NSE.
STT, exchange, GST, SEBI, stamp duty, brokerage.
"""
import config


class ChargeCalculator:
    """Calculates trading charges for NSE equity orders."""

    def __init__(self, is_intraday: bool = True):
        self.is_intraday = is_intraday

    def calculate_total(self, entry_price: float, exit_price: float,
                        quantity: int) -> float:
        """Calculate total charges for a round-trip trade.

        Args:
            entry_price: buy/entry price
            exit_price: sell/exit price
            quantity: number of shares

        Returns:
            Total charges in INR
        """
        buy_value = entry_price * quantity
        sell_value = exit_price * quantity
        turnover = buy_value + sell_value

        charges = {}

        # Brokerage (Rs 20 per order, 2 orders)
        charges["brokerage"] = config.BROKERAGE_PER_ORDER * 2

        # STT
        if self.is_intraday:
            # Intraday: 0.025% on sell side only
            charges["stt"] = sell_value * config.STT_INTRADAY_PCT / 100
        else:
            # Delivery: 0.1% on both sides
            charges["stt"] = turnover * config.STT_DELIVERY_PCT / 100

        # Exchange transaction charges
        charges["exchange"] = turnover * config.EXCHANGE_CHARGES_PCT / 100

        # GST (18% on brokerage + exchange charges)
        charges["gst"] = (charges["brokerage"] + charges["exchange"]) * config.GST_PCT / 100

        # SEBI charges (Rs 10 per crore)
        charges["sebi"] = turnover * config.SEBI_CHARGES_PER_CRORE / 1e7

        # Stamp duty (0.015% on buy side)
        charges["stamp_duty"] = buy_value * config.STAMP_DUTY_PCT / 100

        total = sum(charges.values())
        return round(total, 2)

    def get_breakdown(self, entry_price: float, exit_price: float,
                      quantity: int) -> dict:
        """Get detailed charge breakdown."""
        buy_value = entry_price * quantity
        sell_value = exit_price * quantity
        turnover = buy_value + sell_value

        brokerage = config.BROKERAGE_PER_ORDER * 2

        if self.is_intraday:
            stt = sell_value * config.STT_INTRADAY_PCT / 100
        else:
            stt = turnover * config.STT_DELIVERY_PCT / 100

        exchange = turnover * config.EXCHANGE_CHARGES_PCT / 100
        gst = (brokerage + exchange) * config.GST_PCT / 100
        sebi = turnover * config.SEBI_CHARGES_PER_CRORE / 1e7
        stamp = buy_value * config.STAMP_DUTY_PCT / 100

        total = brokerage + stt + exchange + gst + sebi + stamp

        # P&L
        if entry_price < exit_price:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        net_pnl = gross_pnl - total

        return {
            "buy_value": round(buy_value, 2),
            "sell_value": round(sell_value, 2),
            "turnover": round(turnover, 2),
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange, 2),
            "gst": round(gst, 2),
            "sebi_charges": round(sebi, 2),
            "stamp_duty": round(stamp, 2),
            "total_charges": round(total, 2),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "charges_pct": round(total / turnover * 100, 4) if turnover > 0 else 0,
        }
