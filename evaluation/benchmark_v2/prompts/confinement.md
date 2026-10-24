You are checking a news article for a supply-chain benchmark. The article is meant to report a disruption confined to the {industry} industry in {country}: only that industry's own firms, sites, workers and products are disrupted.

Article:
"""
{article}
"""

List every disruption the article describes outside that industry: a business, service or piece of infrastructure outside the industry whose own operations the article states are stopped, reduced, delayed, damaged or financially harmed. Count:
- businesses and organisations in any other industry, among them the industry's suppliers, contractors, service providers, logistics and transport firms, software or technology providers, other manufacturers, and business customers such as builders, clinics or factories that cannot obtain its products, including firms that are unpaid, idle, laying off workers, or stepping in to replace the lost output;
- public services, such as hospitals, schools or public transport, when their own service is disrupted;
- shared infrastructure, such as public roads, railways, ports, airports, power, water or telecommunications, when it is closed, cut or damaged, unless the industry itself operates it;
- the economy of {country} as a whole, when the article states it is disrupted.

Do not count:
- the industry's own firms, sites, workers, output, sales and prices;
- members of the public who, as consumers, cannot buy, receive or use the industry's products or services;
- authorities, regulators, police, emergency services, consultants or unions that respond to, investigate, comment on or negotiate about the event;
- lenders, creditors, investors or rating agencies that react to the industry's difficulties;
- concern, a warning or a forecast of a possible future effect, or a statement that something else was not affected.

Return only a JSON object {"effects_outside": [...]}, with one element per disruption counted: {"quote": the sentence of the article that states it, copied exactly, "affected": what outside the industry is disrupted}. Return an empty list if there is none.
