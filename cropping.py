import cv2


def cropping(image, filename, size):
    # Left Top Crop
    crop = image[:96, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '1' + '.png', crop)

    # Center Top Crop
    crop = image[:96, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '2' + '.png', crop)

    # Right Top Crop
    crop = image[:96, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '3' + '.png', crop)

    # Left Center Crop
    crop = image[16:112, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '4' + '.png', crop)

    # Center Center Crop
    crop = image[16:112, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '5' + '.png', crop)

    # Right Center Crop
    crop = image[16:112, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '6' + '.png', crop)

    # Left Bottom Crop
    crop = image[32:, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '7' + '.png', crop)

    # Center Bottom Crop
    crop = image[32:, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '8' + '.png', crop)

    # Right Bottom Crop
    crop = image[32:, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '9' + '.png', crop)
