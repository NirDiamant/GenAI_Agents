"""
This script demonstrates how to create a simple MCP server that fetches
the current price of a cryptocurrency using the CoinGecko API.
It uses the FastMCP library to create the server and handle requests.
"""
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Create our MCP server with a descriptive name
mcp = FastMCP("crypto_price_tracker")

# Now let's define our first tool - getting the current price of a cryptocurrency
@mcp.tool()
async def get_crypto_price(crypto_id: str, currency: str = "usd") -> str:
    """
    Get the current price of a cryptocurrency in a specified currency.
    
    Parameters:
    - crypto_id: The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
    - currency: The currency to display the price in (default: 'usd')
    
    Returns:
    - Current price information as a formatted string
    """
    # Construct the API URL
    url = f"{COINGECKO_BASE_URL}/simple/price"
    
    # Set up the query parameters
    params = {
        "ids": crypto_id,
        "vs_currencies": currency
    }
    
    try:
        # Make the API call
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the response
            data = response.json()
            
            # Check if we got data for the requested crypto
            if crypto_id not in data:
                return f"Cryptocurrency '{crypto_id}' not found. Please check the ID and try again."
            
            # Format and return the price information
            price = data[crypto_id][currency]
            return f"The current price of {crypto_id} is {price} {currency.upper()}"
            
    except httpx.HTTPStatusError as e:
        return f"API Error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error fetching price data: {str(e)}"

# You can add more tools here, following the same pattern as above

# Run the MCP server
# This will start the server and listen for incoming requests
if __name__ == "__main__":
    mcp.run()