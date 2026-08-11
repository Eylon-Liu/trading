"""
What an 8-K item code actually means for a thesis.

The item code is the one piece of filing metadata that needs no text parsing —
the filer classified it themselves. But a code and its official title stop
short of the question a reader has, which is "does this matter?".

"5.02 — Director / officer departure or appointment" covers a CEO being fired
and a well-regarded hire arriving, which are opposite signals. So each code
carries three things here: what it usually means, how much weight to give it,
and what to look for in the filing itself.

None of this feeds a score. Item *counts* are factors (see EVENT_MANAGEMENT
and friends); this module is the reading, kept separate so a judgement call
about significance can never leak into a ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# How much a single filing of this type usually deserves.
ROUTINE = 'routine'        # nearly always housekeeping
NOTABLE = 'notable'        # worth a look
MATERIAL = 'material'      # can change a thesis on its own

WEIGHT_ORDER = {ROUTINE: 0, NOTABLE: 1, MATERIAL: 2}


@dataclass(frozen=True)
class ItemMeaning:
    significance: str
    means: str          # what the filing is
    read: str           # how to interpret it
    check: str          # what to look for before acting


MEANINGS: dict[str, ItemMeaning] = {
    '1.01': ItemMeaning(
        NOTABLE,
        'The company signed a material contract — a partnership, supply deal, '
        'credit facility or licence.',
        'Direction is not in the code. A new revenue agreement and a rescue '
        'financing are both filed as 1.01.',
        'Read the exhibit: who the counterparty is, the size, and the term.'),
    '1.02': ItemMeaning(
        MATERIAL,
        'A material agreement was terminated.',
        'Usually negative — a lost customer, a cancelled deal, a pulled credit '
        'line — but occasionally the company walking away from a bad contract.',
        'Whether the company terminated or was terminated, and the revenue '
        'attached to it.'),
    '2.01': ItemMeaning(
        MATERIAL,
        'An acquisition or disposal completed.',
        'Changes the business you are analysing. Every trailing ratio now '
        'mixes the old and new company, and growth figures become '
        'incomparable until a full year has passed.',
        'The price paid, how it was funded, and whether goodwill will now '
        'flatter or depress return on capital.'),
    '2.02': ItemMeaning(
        NOTABLE,
        'Quarterly or annual results were released.',
        'Expected on a schedule, so the filing itself is not news. What '
        'matters is the reaction: this is the event that opens the '
        'post-earnings drift window.',
        'The price move on the day, and whether guidance moved with it.'),
    '2.05': ItemMeaning(
        MATERIAL,
        'Restructuring: layoffs, plant closures, or exiting a business line.',
        'The company is under cost pressure. Can precede a genuine turnaround '
        'or mark the start of a decline; the code cannot tell you which.',
        'The charge size against annual operating income, and whether this is '
        'the first such filing or the third in two years.'),
    '2.06': ItemMeaning(
        MATERIAL,
        'A material asset was written down.',
        'Management has conceded that something is worth less than carried — '
        'often an acquisition that did not work. Book value and any '
        'book-based valuation factor are affected directly.',
        'What was impaired, and whether book-to-market screened well *because* '
        'of the assets now being written off.'),
    '3.01': ItemMeaning(
        MATERIAL,
        'Delisting notice or failure to meet a listing rule.',
        'Serious. Usually follows a price collapse, a late filing, or a '
        'governance failure.',
        'Which rule, and the remediation deadline. This is rarely a value '
        'opportunity.'),
    '4.01': ItemMeaning(
        NOTABLE,
        'The company changed auditor.',
        'Routine on a long cycle, a red flag when sudden — especially if the '
        'outgoing auditor resigned rather than being replaced.',
        'Whether the auditor resigned or was dismissed, and whether there '
        'were disagreements on accounting treatment.'),
    '4.02': ItemMeaning(
        MATERIAL,
        'Previously issued financial statements can no longer be relied upon.',
        'The most serious item on this list. Historical figures are being '
        'restated, which means every fundamental factor computed from them '
        'was computed from numbers the company now disowns.',
        'Which periods and which lines. Treat fundamental scores for this '
        'name as unreliable until the restatement is filed.'),
    '5.02': ItemMeaning(
        NOTABLE,
        'A director or officer arrived or departed.',
        'Direction-blind by design: a respected hire and an abrupt CFO exit '
        'file identically. An abrupt exit with no successor named is the '
        'pattern worth noticing, as is repeated turnover in the same seat.',
        'Whether a successor was named at the same time, and whether the '
        'departure was described as immediate.'),
    '5.07': ItemMeaning(
        ROUTINE,
        'Results of a shareholder vote.',
        'Housekeeping in almost every case. Occasionally informative when '
        'say-on-pay or a director election draws unusual opposition.',
        'Only worth reading if a proposal failed or drew heavy dissent.'),
    '7.01': ItemMeaning(
        ROUTINE,
        'A Regulation FD disclosure — usually an investor presentation.',
        'Rarely news in itself; it is the mechanism for making something '
        'public simultaneously.',
        'The attached exhibit, if guidance appears in it.'),
    '8.01': ItemMeaning(
        ROUTINE,
        'A catch-all for anything the company chose to disclose.',
        'Contents vary enormously — buyback authorisations and dividend '
        'declarations often land here alongside genuine trivia.',
        'The headline of the exhibit; this is the one code where reading is '
        'unavoidable.'),
}


def get(item_code: str) -> ItemMeaning | None:
    return MEANINGS.get((item_code or '').strip())


def significance(item_code: str) -> str:
    m = get(item_code)
    return m.significance if m else ROUTINE


def summarize(events: pd.DataFrame) -> dict:
    """
    A reading of a company's recent filing pattern.

    Counts are only the input; the output is the observation a reader would
    make — repeated management turnover, a restructuring cluster, an
    accounting red flag — plus the routine noise explicitly set aside.
    """
    if events is None or events.empty:
        return {'total': 0, 'material': 0, 'notable': 0, 'routine': 0,
                'observations': [], 'headline': 'No filings in this window.'}

    df = events.copy()
    df['significance'] = df['item_code'].map(significance)
    counts = df['significance'].value_counts().to_dict()
    material = int(counts.get(MATERIAL, 0))
    notable = int(counts.get(NOTABLE, 0))
    routine = int(counts.get(ROUTINE, 0))

    observations: list[str] = []
    by_code = df['item_code'].value_counts().to_dict()

    if by_code.get('4.02'):
        observations.append(
            'Prior financials were declared unreliable (item 4.02). Every '
            'fundamental factor for this name is computed from figures the '
            'company has since disowned — treat its scores as suspect.')
    if by_code.get('2.06'):
        observations.append(
            'A material impairment was booked (item 2.06), which reduces book '
            'value directly. Check whether any value score here rested on the '
            'assets just written down.')
    n_mgmt = by_code.get('5.02', 0)
    if n_mgmt >= 3:
        observations.append(
            f'{n_mgmt} management or board changes in this window. Repeated '
            f'turnover is a recognised instability signal — though the code '
            f'cannot distinguish hires from departures, so read the filings.')
    elif n_mgmt:
        observations.append(
            f'{n_mgmt} management change(s). Normal at this rate; worth a '
            f'glance for whether a successor was named.')
    if by_code.get('2.05'):
        observations.append(
            'Restructuring charges were filed (item 2.05) — the company is '
            'cutting costs. That precedes a turnaround about as often as a '
            'decline.')
    if by_code.get('2.01') or by_code.get('1.02'):
        observations.append(
            'The business changed shape through an acquisition, disposal or '
            'terminated contract. Trailing ratios now blend two different '
            'companies.')
    if by_code.get('3.01'):
        observations.append('A listing-rule problem was disclosed (item 3.01).')

    if not observations:
        observations.append(
            'Nothing structurally unusual — results, disclosures and votes at '
            'the cadence any listed company files them.')

    if material:
        headline = (f'{material} material filing(s) worth reading, '
                    f'{notable} notable, {routine} routine.')
    elif notable:
        headline = (f'No material events. {notable} notable filing(s), '
                    f'{routine} routine.')
    else:
        headline = f'All {routine} filing(s) routine.'

    return {'total': len(df), 'material': material, 'notable': notable,
            'routine': routine, 'observations': observations,
            'headline': headline}
