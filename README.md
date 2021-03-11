# ML_PMT_segmentation
Code to process SK images after segmentation to get feature points


super_module_locator is run on the segmentation output for a ring of images selected using the spreadsheet of drone information which the code also takes as input for reprojection. It can be run automatically or with manual intervention where the user decides the line lablels

adjust_number_files is used to correct any errors in this output instead of running the whole stack again, Its also useful for sorting out the half module, however take care as it only changes the super module number so always need to check the output images to make sure the inter module labelling is correct

find_bolts_and_ellipse does what it says taking in the segmentation files and super module locator output

feature_info_analysis performs cuts on ellipse fit output and produces final points files for use in photogrammetry
