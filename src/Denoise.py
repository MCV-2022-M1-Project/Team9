
import numpy as np
import cv2
import matplotlib.pyplot as plt
import bm3d

class Denoise:

    # Estimate noise
    def estimate_noise(im):
        """
        Estimates if an image contains noise
        From https://www.sciencedirect.com/science/article/abs/pii/S1077314296900600
        Reference https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image

        """
        laplacian_difference_kernel = np.array([[1,-2,1], [-2,4,-2], [1,-2,1]])
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_convolved_sum = np.sum(np.abs(cv2.filter2D(src=im_gray, ddepth=-1, kernel=laplacian_difference_kernel)))
        sigma = np.sqrt(np.pi/2) * (1/(6*(im.shape[0]-2)*(im.shape[1]-2))) * im_convolved_sum
        
        return sigma


    # Method 1. Use simple methods and get whichever works best
    def remove_noise_simple(im):
        """
        Removes the noise of an image using different types of spatial and frequency based filters
        We use non-local means https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf
        """
        noise = Denoise.estimate_noise(im)
        is_noisy = False
        if noise > 4:
            print("Denoising image...")
            is_noisy = True
            # Save noise estimate of the resulting filters
            filter_results = {
                'average': {
                    'psnr': 0,
                    'result': []
                },
                'gaussian': {
                    'psnr': 0,
                    'result': []
                },
                'median': {
                    'psnr': 0,
                    'result': []
                },
                'bilateral': {
                    'psnr': 0,
                    'result': []
                },
                'nlmeans': {
                    'psnr': 0,
                    'result': []
                }
            }
            
            # Average
            average_kernel = (1/9) * np.array([[1,1,1], [1,1,1], [1,1,1]])
            im_average = cv2.filter2D(src=im, ddepth=-1, kernel=average_kernel)
            filter_results['average']['psnr'] = cv2.PSNR(im, im_average) 
            filter_results['average']['result'] = im_average
            
            # Gaussian
            im_gaussian = cv2.GaussianBlur(im, (5,5), 1)
            filter_results['gaussian']['psnr'] = cv2.PSNR(im, im_gaussian) 
            filter_results['gaussian']['result'] = im_gaussian
            
            # Median
            im_median = cv2.medianBlur(im, 3)
            filter_results['median']['psnr'] = cv2.PSNR(im, im_median) 
            filter_results['median']['result'] = im_median
            
            # Bilateral
            im_bilateral = cv2.bilateralFilter(im, 15, 15, 80)
            filter_results['bilateral']['psnr'] = cv2.PSNR(im, im_bilateral) 
            filter_results['bilateral']['result'] = im_bilateral
            
            # Non-local means
            im_nlmeans = cv2.fastNlMeansDenoisingColored(im,None, h=0.55*noise, templateWindowSize=3, searchWindowSize=21)
            filter_results['nlmeans']['psnr'] = cv2.PSNR(im, im_nlmeans) 
            filter_results['nlmeans']['result'] = im_nlmeans
            
            best_filter = ''
            best_psnr = 0
            
            # Higher psnr better method
            for filter in filter_results:
                psnr = filter_results[filter]['psnr']
                if best_filter == '':
                    best_filter = filter
                    best_psnr = psnr
                elif psnr > best_psnr:
                    best_filter = filter
                    best_psnr = psnr
                
            output = filter_results[best_filter]['result']
        else:
            output =  im
        return output, is_noisy
            

    # Method 2. Use BM3D implementation 
    def remove_noise_BM3D(im, max_width=300, max_height=300):
        """
            Removes the noise of an image using BM3D
            https://www.ipol.im/pub/art/2012/l-bm3d/article.pdf
        """
        noise = Denoise.estimate_noise(im)
        is_noisy = False
        if noise > 4:
            print("Denoising image...")
            is_noisy = True
            height, width, _ = im.shape
            
            factor = min(max_width / width, max_height / height)
            im_preprocessed = cv2.resize(im, (int(width * factor), int(height * factor)))
            
            im_preprocessed = (1/255) * im.astype(np.float64)
            im_denoised = bm3d.bm3d(im_preprocessed, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            
            im_denoised = (im_denoised*255).astype(np.uint8)
            output =  im_denoised
        else:
            output =  im
        return output, is_noisy
        
