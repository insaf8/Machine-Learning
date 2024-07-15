import time
from PIL import Image, ImageChops
from appium import webdriver
import cv2
import numpy as np
from selenium.webdriver.common.by import By

from appium.webdriver.common.mobileby import MobileBy
from selenium.webdriver.support.ui import WebDriverWait
from appium.webdriver.common.touch_action import TouchAction
from selenium.webdriver.support import expected_conditions as EC

def find_overlapping_region_sift(image1, image2, threshold=0.75):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Check if enough good matches are found
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get the corners of the overlapping region in the first image
        h, w = gray1.shape
        corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)

        return transformed_corners

    return None

def find_difference_start_height(image1_path, image2_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Convert images to grayscale for better comparison
    image1_gray = image1.convert("L")
    image2_gray = image2.convert("L")

    # Compute the absolute difference between the two images
    diff = ImageChops.difference(image1_gray, image2_gray)

    # Get the bounding box of the non-zero regions in the difference image
    bbox = diff.getbbox()

    # If bbox is None, the images are identical
    if bbox is None:
        return 0, image1_gray.height

    # Return the top (y-coordinate) or bottom (height - y-coordinate) of the bounding box
    return bbox[1], bbox[3]
def cut_at_horizontal_line(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to highlight the thin white line
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Perform morphological operations to enhance the line
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Start scanning from the bottom row
    for i in range(binary.shape[0] - 1, -1, -1):
        # Check if the current row contains only white pixels
        if not np.any(binary[i]):
            # If found, cut the image at this row
            image_cut = image[:i, :]
            return image_cut

def stitch_images_vertically(count, output_path, crop_top, crop_bottom, blend_alpha, footer_overlap=100):
    images = []
    for i in range(count):
        image_path = f'screenshot-{i}.png'
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            print(f"Warning: Skipping {image_path} as it could not be loaded or is empty.")
            continue    
 # Your existing logic to handle cropping and appending images
        if i == 0:
            # Do not crop the top for the first image
            cropped_image = image[:crop_bottom-footer_overlap, :]
        elif i + 1 == count:
            # Last image
            cropped_image = image[crop_top:crop_bottom, :]
        else:
            # Middle images
            cropped_image = image[crop_top:crop_bottom-footer_overlap, :]
        
        # Before calling find_overlapping_region_sift, ensure both images are valid
        if i != 0 and images[-1] is not None and cropped_image is not None:
            overlapping_region = find_overlapping_region_sift(images[-1], cropped_image)
            if overlapping_region is not None:
                height_in_pixels = int(overlapping_region[1][0][1])
                cropped_image = cropped_image[height_in_pixels:cropped_image.shape[0], :]
        
        images.append(cropped_image)

    if not images:
        print("No valid images were loaded. Exiting.")
        return

    target_width = min(img.shape[1] for img in images if img is not None and img.size > 0)
    images = [cv2.resize(img, (target_width, img.shape[0])) for img in images if img is not None and img.size > 0]

    total_height = sum(img.shape[0] for img in images)

    stitched_image = np.ones((total_height, target_width, 3), dtype=np.uint8) * 255

    current_height = 0
    for img in images:
        stitched_image[current_height:current_height + img.shape[0], :] = img
        current_height += img.shape[0]
    stitched_image = cut_at_horizontal_line(stitched_image)
    cv2.imwrite(output_path, stitched_image)


desired_cap = {
    "platformName": "Android",
    "deviceName": "Galaxy A70",
    "platformVersion": "10",
    # "appPackage":"net.one97.paytm",
    # "appActivity":"net.one97.paytm.app.LauncherActivity",
    # "appPackage": "com.drx.uat",
    # "appActivity": "com.drx.uat/com.drx.MainActivity",
    "noReset": True  # Prevents clearing the app data upon startup
    # Additional capabilities can be added here if needed
}
class YourPageObjectClass:
    def __init__(self, driver):
        self.driver = driver

    def scroll_down_until_resource_id_count(self, resource_id,  max_swipes=20):
        element_count = 0
        self.swipe_down()
        time.sleep(1)
        current_count = self.get_resource_id_count(resource_id)
        element_count += current_count

        return element_count

    def get_resource_id_count(self, resource_id):
        # Find all elements matching the resource ID
        elements = self.driver.find_elements(By.XPATH, resource_id)
        # Return the number of elements found
        return len(elements)

    def swipe_down(self):
        screen_size = self.driver.get_window_rect()
        scroll_start_y = int(screen_size['height'] * 0.8)
        scroll_end_y = int(screen_size['height'] * 0.5)
        self.driver.swipe(screen_size['width'] // 2, scroll_start_y, screen_size['width'] // 2, scroll_end_y, 1500)
        time.sleep(7)


url = "http://localhost:4723/wd/hub"
driver = webdriver.Remote(desired_capabilities=desired_cap, command_executor=url)
page_object = YourPageObjectClass(driver)

try:
    driver.activate_app('com.guidesly')
    time.sleep(5)

    count = 0
    previous_screenshot = None
    changes_detected = True
    screenshot_path = f'screenshot-{count}.png'
    driver.save_screenshot(screenshot_path)
   
    # Update the while loop condition to use scroll_down_until_resource_id_count
    while page_object.scroll_down_until_resource_id_count('//android.widget.ScrollView/android.view.ViewGroup/android.view.ViewGroup/android.view.ViewGroup[1]') <=1:
        screenshot_path = f'screenshot-{count+1}.png'
        driver.save_screenshot(screenshot_path)
        current_screenshot = Image.open(screenshot_path)

        if previous_screenshot:
            diff = ImageChops.difference(previous_screenshot.convert("L"), current_screenshot.convert("L"))
            if not diff.getbbox():  # No changes detected
                changes_detected = False
                print("No significant changes detected in scroll. Ending capture.")
                break

        previous_screenshot = current_screenshot

        print('Scrolling Page %s' % count)
        count += 1
    counts=count

finally:
    start_height_top, start_height_bottom = find_difference_start_height('screenshot-1.png', 'screenshot-2.png')

    output_path = 'output.png'

    footer_overlap = 100  # This is an example value. Adjust based on your needs.

    # Ensure to pass the footer_overlap parameter.
    stitch_images_vertically(counts, output_path, start_height_top, start_height_bottom, 1, footer_overlap)

    driver.quit()
