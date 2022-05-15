from splinter import Browser
from pathlib import Path


def collect_image(url):
    browser = Browser(headless=True)

    width = 1024
    height = 3 * width // 4

    browser.driver.set_window_size(width, height)
    print("Browser size", browser.driver.get_window_size())
    browser.visit(url)
    image_path = (Path("./") / f"collected_images/img_{url.split('//')[-1]}_").absolute()
    browser.screenshot(
        str(image_path)
    )
    browser.quit()
