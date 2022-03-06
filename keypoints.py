"""
Post-processing of PVNet output model into keypoints
"""

from stringprep import in_table_a1
import torch

"""
findKeypoints() takes image segmentation and keypoint vector, and ouputs set of hypotheses 

"""
def findKeypoints(segmentationMap: torch.Tensor,
        keypointVectorMap: torch.Tensor,
        class_labels: list(),
        device = 'cpu',
        ):
    
    vote_threshold = 0.99 # Threshold for pixel to vote for a hypothesis
    confidence = 0.99  # RANSAC termination threshold
    max_iterations = 20     # Iterations of Ransac if confidence not achieved
    num_hypotheses = 512    # Number of keypoint hypotheses per iteration
    

    print(segmentationMap.size())
    b, num_classes, h, w = segmentationMap.size()
    num_classes -= 1
    num_classes = int(num_classes)
    num_vertices = int(keypointVectorMap.size(1) / num_classes / 2)

    for bi in range(0,b):
        
        # topk(1) returns (max_value, idx_of_channel_with_max_value) 
        # [C,H,W] => ([1,H,W], [1,H,W])
        _ , singleClassMap = torch.topk(segmentationMap[bi], 1, dim=0)

        # [num_class*num_keypoint*2,H,W]
        singleKeyVectorMap = keypointVectorMap[bi]  
        
        # Generate hypothesis for each class label
        for c in class_labels:
            in_class_mask = singleClassMap == c
            inClass = in_class_mask.nonzero() 
            print(inClass)
            
            # Return keypoint vectors for every pixel that's in class
            # [pixels_in_class, 2]
            vectorPtsInClass = torch.masked_select(
                # Only get keypoint vectors for the class of interest
                singleKeyVectorMap[c.item()*2*num_vertices:(c.item()+1)*2*num_vertices,:,:],
                # Only get vectors for points where pixel is in class 
                in_class_mask).reshape(inClass.size(0),num_vertices,2)

            return(vectorPtsInClass)

"""
generate num_hypotheses for each keypoint by randomly sampling points in vectorLookup which are 
assumed to all be within the class for that batch

"""

# def _generate_hypotheses(num_hypotheses,
        
#         vectorLookup,
#         )