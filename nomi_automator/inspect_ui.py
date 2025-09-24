import asyncio
from playwright.async_api import async_playwright
from config import NOMI_URL, HEADLESS

async def inspect_ui():
    """Inspect the Nomi.ai UI to identify chat elements"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page()
        await page.goto(NOMI_URL)
        await page.wait_for_load_state('networkidle')

        # Take a screenshot for visual inspection
        await page.screenshot(path='nomi_ui_screenshot.png')
        print("Screenshot saved as nomi_ui_screenshot.png")

        # Get page title
        title = await page.title()
        print(f"Page title: {title}")

        # Try to find common chat elements
        # Messages container
        messages = await page.query_selector_all('[class*="message"], [class*="chat"], [data-testid*="message"]')
        print(f"Found {len(messages)} potential message elements")

        # Input field
        inputs = await page.query_selector_all('input[type="text"], textarea, [contenteditable]')
        print(f"Found {len(inputs)} potential input elements")

        # Send button
        buttons = await page.query_selector_all('button, [role="button"]')
        print(f"Found {len(buttons)} potential button elements")

        # Print some HTML for inspection
        content = await page.content()
        with open('page_content.html', 'w') as f:
            f.write(content)
        print("Page content saved as page_content.html")

        # Wait for user to inspect
        input("Press Enter to close browser...")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(inspect_ui())