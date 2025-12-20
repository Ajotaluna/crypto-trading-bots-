# Fortress Bot V6 "The Sniper": Mechanics of the Perfect Trade

The user's goal is to hit the daily target (3%) with **one high-quality trade** if possible, rather than churning volume. This requires shifting from "Fixed Sizing" to "Risk-Based Sizing" and implementing "Perfect Tracking".

## 1. The Math of "The One Shot"
To gain 3% Equity in one trade without gambling:
*   **Risk per Trade:** 1% of Equity.
*   **Target Risk/Reward (R:R):** 1:3.
*   **Math:** If we risk 1% and win 3x that risk, we gain 3% Equity.
*   **Requirement:** We need precise position sizing based on *Volatility*.

## 2. Implementation: The Risk Vault (Sizing)
We stop using `CAPITAL_PER_TRADE_PCT`. We switch to `RISK_PER_TRADE_PCT`.
*   **Formula:** `Position Size = (Account Balance * Risk %) / (Entry - Stop Loss)`
*   **Constraint:** Max Position Size cap (to avoid overexposure on tight stops).

### Changes Required:
1.  **Config**: Add `RISK_PER_TRADE_PCT = 1.0` (Risk 1% of account).
2.  **Config**: Add `TARGET_RISK_REWARD = 3.0` (Aim for 3% gain).
3.  **MarketData**: Add `calculate_position_size(entry, sl)` method.
4.  **Main**: Use this new sizing logic in `execute_trade`.

## 3. Implementation: The Harvester (Perfect Tracking)
"Perfect Tracking" means locking in wins and letting runners run.
*   **The Guard Rail:** Move Stop Loss to **Break Even** as soon as price hits 1R (1x Risk).
*   **The Harvest:** Take 50% Profit at 2R (Secure the bag).
*   **The Runner:** Let the remaining 50% run with the ATR Trailing Stop (Bloodhound) to catch the "Home Run".

### Changes Required:
1.  **Main (`manage_positions`)**:
    *   Track `R_multiple` (current profit / initial risk).
    *   Trigger `BE_LOCK` (Break Even) at 1R.
    *   Trigger `PARTIAL_TP` at 2R.

## 4. Execution Plan
1.  **Config Update:** Define Risk Parameters.
2.  **MarketData Update:** Implement Sizing Math.
3.  **Main Update:** Implement BE Lock and Partial TP.
