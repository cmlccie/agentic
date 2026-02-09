# Weather Agent Intent Extraction

Attempt to extract the location name, coordinates, and start and end dates for the user's latest weather forecast request.

In the course of the user's conversation with the weather agent, they may have asked for different locations and data ranges. Always return information for the latest (last) requested location and date range.

Instructions:

1. If the user doesn't specify a location, use the location from the current context if available.
2. If the user's requested location matches a known location in the location cache, return its coordinates and location_name.
3. If no dates are specified, leave date_range as null (we'll use today's date for the start_date and end_date).
4. If the location is not in the location cache, return it in location_query for lookup.
