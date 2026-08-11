"""Shared visual language — colors, table styles, and the page shell CSS."""

from __future__ import annotations

import config

# ─────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────

BG = '#0f1117'
PANEL = '#1a1d2e'
PANEL_ALT = '#252836'
BORDER = 'rgba(255,255,255,0.08)'
TEXT = '#e0e0e0'
MUTED = '#8b93a7'

ACCENT = '#38ef7d'
ACCENT_DEEP = '#11998e'
GRADIENT = f'linear-gradient(135deg,{ACCENT_DEEP} 0%,{ACCENT} 100%)'

POS = '#38ef7d'
NEG = '#ff6b6b'
WARN = '#ffd93d'
INFO = '#00d4ff'

SECTOR_COLORS = config.SECTOR_COLORS

HORIZON_COLOR = {'long': '#00d4ff', 'mid': '#ffd93d'}


# ─────────────────────────────────────────────
# PLOTLY
# ─────────────────────────────────────────────

def plot_layout(title: str = '', height: int | None = None, **kw) -> dict:
    """Consistent dark-theme layout for every figure."""
    layout = {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': TEXT, 'size': 11},
        'margin': {'l': 50, 'r': 20, 't': 46 if title else 16, 'b': 44},
        'hoverlabel': {'bgcolor': PANEL_ALT, 'font': {'color': TEXT}},
        'xaxis': {'showgrid': False, 'zeroline': False,
                  'linecolor': 'rgba(255,255,255,0.15)'},
        'yaxis': {'showgrid': True, 'griddash': 'dash',
                  'gridcolor': 'rgba(255,255,255,0.10)', 'zeroline': False},
        'legend': {'bgcolor': 'rgba(0,0,0,0)', 'font': {'size': 10}},
    }
    if title:
        layout['title'] = {'text': title, 'x': 0.5, 'xanchor': 'center',
                           'font': {'size': 13, 'color': TEXT}}
    if height:
        layout['height'] = height
    layout.update(kw)
    return layout


def empty_figure(message: str = 'Run an analysis to populate this view.') -> dict:
    """Placeholder figure carrying an explanatory message."""
    return {
        'data': [],
        'layout': plot_layout(height=280, annotations=[{
            'text': message, 'x': 0.5, 'y': 0.5, 'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 13, 'color': MUTED},
        }]),
    }


# ─────────────────────────────────────────────
# DATATABLE
# ─────────────────────────────────────────────

TABLE_HEADER = {
    'background': GRADIENT, 'color': '#000', 'fontWeight': '700',
    'border': f'1px solid {BG}', 'textAlign': 'left',
    'padding': '10px 12px', 'fontSize': '0.78rem',
    'textTransform': 'uppercase', 'letterSpacing': '0.4px',
}

# One line per row. With `whiteSpace: normal` a single wordy cell — a reason
# string, a long sector name — wrapped to three lines and set the height for
# every other column, so fifteen results filled several screens. Overflow is
# clipped with an ellipsis and the full value is available on hover.
TABLE_CELL = {
    'backgroundColor': PANEL, 'color': TEXT,
    'border': '1px solid rgba(255,255,255,0.05)',
    'padding': '7px 12px', 'fontSize': '0.82rem',
    'fontFamily': '-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif',
    'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis',
    'maxWidth': '210px', 'height': '34px', 'textAlign': 'left',
}

# Explicit widths for the columns that appear across pages. Anything not
# listed and numeric gets a right-aligned default; anything else flexes.
COLUMN_WIDTHS = {
    'rank': '52px', 'ticker': '72px', 'sector': '130px',
    'score': '74px', 'composite': '74px', 'signal': '156px',
    'why': '160px', 'reasons': '160px',
    'entry_type': '96px', 'stop_basis': '104px',
    'insider': '150px', 'role': '140px', 'side': '64px',
    'txn_date': '96px', 'entry_date': '96px', 'exit_date': '96px',
    'Factor': '190px', 'Family': '110px', 'Direction': '120px',
}

TABLE_CONDITIONAL = [
    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(255,255,255,0.02)'},
    {'if': {'filter_query': '{signal} contains "Buy" || {signal} contains "Enter"'},
     'backgroundColor': 'rgba(56,239,125,0.07)'},
    {'if': {'filter_query': '{signal} contains "Avoid" || {signal} contains "Trim"'},
     'backgroundColor': 'rgba(255,107,107,0.07)'},
    {'if': {'state': 'active'},
     'backgroundColor': 'rgba(17,153,142,0.20)', 'border': f'1px solid {ACCENT}'},
]


def signal_style(column: str = 'signal') -> list[dict]:
    """Color the signal column by direction."""
    return [
        {'if': {'filter_query': f'{{{column}}} contains "Buy" || '
                                f'{{{column}}} contains "Enter"',
                'column_id': column},
         'color': POS, 'fontWeight': '700'},
        {'if': {'filter_query': f'{{{column}}} contains "Avoid" || '
                                f'{{{column}}} contains "Trim"',
                'column_id': column},
         'color': NEG, 'fontWeight': '700'},
        {'if': {'filter_query': f'{{{column}}} contains "Hold" || '
                                f'{{{column}}} contains "Watch"',
                'column_id': column},
         'color': WARN},
    ]


# ─────────────────────────────────────────────
# PAGE SHELL
# ─────────────────────────────────────────────

INDEX_CSS = f"""
body {{ background-color:{BG} !important; color:{TEXT}; }}
::-webkit-scrollbar {{ width:10px; height:10px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:#333a52; border-radius:5px; }}
::-webkit-scrollbar-thumb:hover {{ background:#454e6b; }}

.card {{ border:1px solid {BORDER} !important; background:{PANEL} !important;
        border-radius:10px !important; }}
.card-header {{ background:rgba(17,153,142,0.10) !important;
               border-bottom:1px solid rgba(17,153,142,0.28) !important;
               border-radius:10px 10px 0 0 !important; }}

.nav-tabs {{ border-bottom:1px solid {BORDER}; }}
.nav-tabs .nav-link {{ color:{MUTED} !important; border:none !important;
                      background:transparent !important; font-weight:600;
                      padding:11px 20px; }}
.nav-tabs .nav-link:hover {{ color:{TEXT} !important; }}
.nav-tabs .nav-link.active {{ color:{ACCENT} !important;
                             border-bottom:2px solid {ACCENT} !important;
                             background:rgba(17,153,142,0.08) !important; }}

/* Dash 4.x dropdowns.

   The root element is a <button class="dash-dropdown"> that ships a white
   background from the component's own stylesheet, while every text node
   inside inherits white — so on a dark theme the control renders as a blank
   white box with invisible text. Styling only the inner .dash-dropdown-trigger
   does not fix it; the button itself has to be overridden, and the chevron
   inherits a dark navy that also has to be lightened.

   The legacy .Select-* selectors below are kept so the theme still works on
   Dash 2.x. */
button.dash-dropdown, .dash-dropdown {{
    background:{PANEL_ALT} !important;
    background-color:{PANEL_ALT} !important;
    border:1px solid #3d4166 !important;
    border-radius:6px !important;
    color:{TEXT} !important;
    min-height:38px;
}}
button.dash-dropdown:hover {{ border-color:{ACCENT_DEEP} !important; }}
button.dash-dropdown *, .dash-dropdown-wrapper * {{ color:{TEXT} !important; }}
button.dash-dropdown svg, button.dash-dropdown svg * {{
    color:{MUTED} !important; fill:{MUTED} !important; stroke:{MUTED} !important; }}
.dash-dropdown-trigger, .dash-dropdown-grid-container {{
    background:transparent !important; border-color:#3d4166 !important;
    color:{TEXT} !important; }}
.dash-dropdown-value, .dash-dropdown-value-item,
.dash-dropdown-trigger span, .dash-dropdown-trigger div {{
    color:{TEXT} !important; }}
.dash-dropdown-placeholder {{ color:#6b7280 !important; }}
.dash-dropdown-menu, .dash-dropdown-options, .dash-dropdown-listbox {{
    background:{PANEL_ALT} !important; border:1px solid #3d4166 !important;
    z-index:9999 !important; }}
.dash-dropdown-option, .dash-dropdown-option * {{ color:{TEXT} !important; }}
.dash-dropdown-option:hover, .dash-dropdown-option[aria-selected="true"] {{
    background:#3d4166 !important; color:#fff !important; }}
.dash-dropdown-clear, .dash-dropdown-arrow {{ color:{MUTED} !important; }}

/* Legacy Dash 2.x dropdown markup */
.Select-control, .Select-menu-outer {{ background:{PANEL_ALT} !important;
                                      border-color:#3d4166 !important; }}
.Select-value-label, .Select-placeholder {{ color:{TEXT} !important; }}
.Select-menu-outer {{ z-index:9999 !important; }}
.VirtualizedSelectOption {{ background:{PANEL_ALT} !important; color:{TEXT} !important; }}
.VirtualizedSelectFocusedOption {{ background:#3d4166 !important; color:#fff !important; }}
.Select-arrow-zone .Select-arrow {{ border-top-color:{MUTED} !important; }}

/* Date picker */
.DateInput, .DateInput_input, .SingleDatePickerInput, .DateRangePickerInput {{
    background:{PANEL_ALT} !important; color:{TEXT} !important;
    border-color:#3d4166 !important; }}
.DateInput_input {{ font-size:0.86rem !important; padding:7px 9px !important; }}
.SingleDatePicker_picker, .DayPicker, .CalendarMonth, .CalendarMonthGrid {{
    background:{PANEL} !important; color:{TEXT} !important; }}
.CalendarDay__default {{ background:{PANEL} !important; color:{TEXT} !important;
    border-color:#2a2e42 !important; }}
.CalendarDay__selected, .CalendarDay__selected:hover {{
    background:{ACCENT_DEEP} !important; color:#fff !important; }}
.DayPickerNavigation_button {{ background:{PANEL_ALT} !important;
    border-color:#3d4166 !important; }}
.CalendarMonth_caption, .DayPicker_weekHeader {{ color:{TEXT} !important; }}

/* Sliders. Dash 4.x renders `.dash-slider-*`; the older `.rc-slider-*` classes
   are kept for compatibility. Without the rules below the value tooltip is
   white-on-white and the tick marks are near-black on the dark panel. */
.rc-slider-track, .dash-slider-track {{ background:{ACCENT_DEEP} !important; }}
.rc-slider-rail, .dash-slider-rail {{ background:#2a2e42 !important; }}
.rc-slider-handle, .dash-slider-thumb {{
    border-color:{ACCENT} !important; background:{ACCENT} !important; }}
.rc-slider-mark-text, .dash-slider-mark {{
    color:{MUTED} !important; font-size:10px !important; }}
.rc-slider-mark-text-active, .dash-slider-mark-within-selection {{
    color:{TEXT} !important; }}
/* Dash 4.x renders a numeric entry box beside each slider; unstyled it is a
   bare white rectangle on the dark panel. */
.dash-input-container, .dash-range-slider-input, .dash-slider-input {{
    background:{PANEL_ALT} !important;
    color:{TEXT} !important;
    border:1px solid #3d4166 !important;
    border-radius:5px !important;
    font-size:0.8rem !important;
    text-align:center; }}
.dash-input-container:focus, .dash-range-slider-input:focus {{
    outline:none !important; border-color:{ACCENT} !important; }}

.dash-slider-tooltip {{
    background:{ACCENT_DEEP} !important;
    color:#fff !important;
    border:1px solid {ACCENT} !important;
    border-radius:4px !important;
    font-size:11px !important; font-weight:700; padding:1px 6px !important; }}
.dash-slider-tooltip * {{ color:#fff !important; }}

.form-control, .form-control:focus {{ background:{PANEL_ALT} !important;
    border:1px solid #3d4166 !important; color:{TEXT} !important;
    box-shadow:none !important; }}
.form-control::placeholder {{ color:#5a6178 !important; }}

#run-button:hover:not(:disabled) {{ transform:translateY(-2px);
    box-shadow:0 8px 28px rgba(17,153,142,0.55) !important; }}
#run-button {{ transition:all .22s ease; }}
#run-button:disabled {{ opacity:.5; cursor:not-allowed; }}

.dash-spreadsheet-container .dash-spreadsheet-inner td,
.dash-spreadsheet-container .dash-spreadsheet-inner th {{ border-color:
    rgba(255,255,255,0.05) !important; }}
.dash-table-container .previous-next-container {{ color:{MUTED}; }}

/* Radio / checkbox labels — Dash renders bare <label> elements that inherit
   Bootstrap's light-theme body color, i.e. near-black on a dark page. */
label.dash-options-list-option, .dash-options-list-option,
.dash-radio-items label, .dash-checklist label,
.form-check-label {{ color:{TEXT} !important; }}
label.dash-options-list-option.selected {{
    color:{ACCENT} !important; font-weight:600; }}
input[type="radio"], input[type="checkbox"] {{ accent-color:{ACCENT_DEEP}; }}

/* Accordions — Bootstrap's expanded state defaults to a light blue that is
   unreadable on a dark page, so both states are pinned explicitly. */
.accordion, .accordion-item {{
    background:transparent !important;
    border-color:{BORDER} !important; }}
.accordion-button {{
    background:{PANEL_ALT} !important;
    color:{TEXT} !important;
    font-size:0.87rem; font-weight:600;
    box-shadow:none !important; }}
.accordion-button:not(.collapsed) {{
    background:{PANEL_ALT} !important;
    color:{ACCENT} !important;
    border-bottom:1px solid {BORDER} !important; }}
.accordion-button:hover {{ background:#252a44 !important; }}
.accordion-button:focus {{
    border-color:{BORDER} !important; box-shadow:none !important; }}
.accordion-button::after, .accordion-button:not(.collapsed)::after {{
    filter:invert(72%) sepia(18%) saturate(420%) hue-rotate(190deg); }}
.accordion-body {{
    background:{PANEL} !important; color:{TEXT} !important;
    font-size:0.85rem; }}
.accordion-flush .accordion-item {{
    border-top:0 !important; border-left:0 !important; border-right:0 !important;
    border-bottom:1px solid {BORDER} !important; }}

.badge {{ padding:.42em .8em !important; font-weight:600; }}
.metric-value {{ font-size:1.5rem; font-weight:800; letter-spacing:-.5px; }}
.metric-label {{ font-size:.7rem; color:{MUTED}; text-transform:uppercase;
                letter-spacing:.5px; }}
a {{ color:{ACCENT}; }}
"""

INDEX_STRING = f'''<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{INDEX_CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
  </body>
</html>
'''
