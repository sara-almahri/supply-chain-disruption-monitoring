You are checking a news article for a supply-chain benchmark. Read the article and record what it reports, using the definitions below. Record only what the article states; add nothing from your own knowledge.

Article:
"""
{article}
"""

1. Ongoing disruption. Decide whether the article reports a disruption that is under way: an event that is currently stopping or reducing production, supply, transport or trade. Set "ongoing_disruption" to false when the article reports no disruption (for example company results, investment, appointments, product launches or routine trade figures), a disruption that has ended or been resolved (a strike that has ended, a port that has reopened, sanctions that have been lifted), a past disruption described as history, or only a forecast, warning or threat of a future disruption.

2. Type. If there is an ongoing disruption, give exactly one of these labels:
   - "Geopolitical": wars, sanctions, political instability, territorial disputes
   - "Trade Policy": tariffs, trade wars, import or export restrictions, embargoes
   - "Natural Disaster": earthquakes, floods, hurricanes, volcanic eruptions, wildfires
   - "Labour Strike": worker strikes, union disputes, labour shortages, walkouts
   - "Economic Crisis": recessions, currency collapse, hyperinflation, financial crises
   - "Company Bankruptcy": corporate insolvency, liquidation, financial distress
   - "Cyber Attack": ransomware, data breaches, IT disruptions
   - "Pandemic": disease outbreaks, health emergencies
   - "Other": a disruption that fits none of the labels above
   Use the most specific label: a labour strike is not an economic crisis, and a trade war is not geopolitical.
   If there is no ongoing disruption, the type is "None".

3. Countries, industries and companies. List only what the article states is disrupted. Do not add countries, industries or companies from your own knowledge, and do not list those mentioned only as background, as sources of comment, as customers or as unaffected.
   - countries: each country in which the disruption is taking place, by its English name (for example "USA", "United Kingdom", "South Korea"). If the article names a region, list the countries it names within it.
   - industries: each disrupted industry, using a name from this list when one applies: "Aerospace & Defense", "Automotive", "Cable & Satellite", "Chemicals", "Construction Materials", "Diversified Industrials", "Electric Utilities", "Electrical Equipment", "Engineering & Construction", "Gas & Water Utilities", "Industrial Intermediate Prod", "Machinery", "Medical Equipment & Devices", "Metals & Mining", "Oil & Gas Producers", "Oil & Gas Services & Equip", "Renewable Energy", "Semiconductors", "Steel", "Technology Hardware", "Technology Services", "Telecommunications", "Transportation & Logistics", "Transportation Equipment". For a disrupted industry not in the list, write "other".
   - companies: each company the article names as disrupted, with the name the article uses.
   If there is no ongoing disruption, the three lists are empty.

4. Scope. "industries" when the article confines the disruption to the industries you listed (for example a strike at copper mines, a cyber attack on chip makers, export controls on steel); "country" when the disruption affects economic activity in the listed countries generally (for example a war, an earthquake, a national lockdown, a currency collapse, a general strike). The scope is also "country" when the article describes effects outside the industry where the event takes place: on businesses in other industries, on public services such as hospitals or public transport, or on shared infrastructure such as public roads, railways, ports, airports, or power or water supply. If there is no ongoing disruption, the scope is "country".

Return only a JSON object with the keys ongoing_disruption (true or false), type, scope, countries, industries and companies.
