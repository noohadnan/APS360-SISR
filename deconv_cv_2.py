from __future__ import print_function

import numpy as np
import cv2
import math
import sys, getopt
from skimage.metrics import structural_similarity as ssim  # for SSIM calculation

from metrics import calculate_psnr, calculate_ssim

def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d, d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w_mask = np.minimum(np.float32(dist)/d, 1.0)
    return img * w_mask + img_blur * (1 - w_mask)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.CV_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern

if __name__ == '__main__':
    print(__doc__)
    opts, args = getopt.getopt(sys.argv[1:], '', ['circle', 'angle=', 'd=', 'snr='])
    opts = dict(opts)
    try:
        fn = args[0]
    except:
        fn = 'SampleData/text.jpg'

    win = 'deconvolution'

    # Load image in grayscale and color
    img_bw = cv2.imread(fn, 0)
    img_rgb = cv2.imread(fn, 1)
    if img_bw is None and img_rgb is None:
        print('Failed to load image:', fn)
        sys.exit(1)

    # Normalize images to [0,1]
    img_rgb = np.float32(img_rgb) / 255.0
    img_bw = np.float32(img_bw) / 255.0

    # Display the input image (converted to 8-bit for display)
    cv2.imshow('input', (img_rgb * 255).astype(np.uint8))

    # Split color channels and blur their edges
    img_r = blur_edge(img_rgb[..., 0])
    img_g = blur_edge(img_rgb[..., 1])
    img_b = blur_edge(img_rgb[..., 2])

    # Compute the DFT for each channel
    IMG_R = cv2.dft(img_r, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_G = cv2.dft(img_g, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_B = cv2.dft(img_b, flags=cv2.DFT_COMPLEX_OUTPUT)

    defocus = '--circle' in opts

    def update(_):
        # Get trackbar positions
        ang = np.deg2rad(cv2.getTrackbarPos('angle', win))
        d = cv2.getTrackbarPos('d', win)
        noise = 10 ** (-0.1 * cv2.getTrackbarPos('SNR (db)', win))

        # Choose PSF type based on user option
        if defocus:
            psf = defocus_kernel(d)
        else:
            psf = motion_kernel(ang, d)
        cv2.imshow('psf', psf)

        # Normalize PSF and pad it to image size
        psf /= psf.sum()
        psf_pad = np.zeros_like(img_bw)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf

        # Compute DFT of PSF
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
        PSF2 = (PSF ** 2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[..., np.newaxis]

        # Apply Wiener deconvolution on each channel
        RES_R = cv2.mulSpectrums(IMG_R, iPSF, 0)
        RES_G = cv2.mulSpectrums(IMG_G, iPSF, 0)
        RES_B = cv2.mulSpectrums(IMG_B, iPSF, 0)

        res_r = cv2.idft(RES_R, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        res_g = cv2.idft(RES_G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        res_b = cv2.idft(RES_B, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Merge channels back into a color image
        res_rgb = np.zeros_like(img_rgb)
        res_rgb[..., 0] = res_r
        res_rgb[..., 1] = res_g
        res_rgb[..., 2] = res_b

        # Adjust for DFT shift (rolling the image)
        res_rgb = np.roll(res_rgb, -kh // 2, axis=0)
        res_rgb = np.roll(res_rgb, -kw // 2, axis=1)

        # Compute quality metrics (PSNR & SSIM)
        # For a fair comparison we convert both images to grayscale.
        # Convert normalized images to 8-bit then back to float in [0,1]
        input_gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        res_gray = cv2.cvtColor((np.clip(res_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        input_gray = input_gray.astype(np.float32) / 255.0
        res_gray = res_gray.astype(np.float32) / 255.0

        psnr_value = calculate_psnr(input_gray, res_gray)
        ssim_value = calculate_ssim(input_gray, res_gray)

        # Overlay the metrics on the deconvolved image for display.
        display_img = (np.clip(res_rgb, 0, 1) * 255).astype(np.uint8)
        text = f"PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.3f}"
        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.imshow(win, display_img)

    # Set up windows and trackbars
    cv2.namedWindow(win)
    cv2.namedWindow('psf', 0)
    cv2.createTrackbar('angle', win, int(opts.get('--angle', 135)), 180, update)
    cv2.createTrackbar('d', win, int(opts.get('--d', 22)), 50, update)
    cv2.createTrackbar('SNR (db)', win, int(opts.get('--snr', 25)), 50, update)
    update(None)

    # Main loop: update on key presses
    while True:
        ch = cv2.waitKey() & 0xFF
        if ch == 27:  # ESC key to exit
            break
        if ch == ord(' '):
            defocus = not defocus
            update(None)
