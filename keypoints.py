"""
Post-processing of PVNet output model into keypoints
"""
from math import hypot
from numpy import single
import torch

"""
findKeypoints() takes image segmentation and keypoint vector, and ouputs set of hypotheses 

"""
def findKeypoints(segmentationMap: torch.Tensor,
        keypointVectorMap: torch.Tensor,
        class_labels: list(),
        device = 'cpu',
        vote_threshold = 0.999, # Threshold for pixel to vote for a hypothesis
        confidence = 0.99,      # RANSAC termination threshold
        max_iterations = 20,    # Iterations of Ransac if confidence not achieved
        num_hypotheses = 512,   # Number of keypoint hypotheses per iteration        
        ):
    
    device = torch.device("cuda:0" if device == "cuda" else "cpu")


    
    # print(segmentationMap.size())
    b, num_classes, h, w = segmentationMap.size()
    num_classes -= 1
    num_classes = int(num_classes)
    num_vertices = int(keypointVectorMap.size(1) / num_classes / 2)

    for bi in range(0,b):
        
        # topk(1) returns (max_value, idx_of_channel_with_max_value) 
        # [C,H,W] => ([1,H,W], [1,H,W])
        _ , singleClassMap = torch.max(segmentationMap[bi], dim=0)
        print(singleClassMap.size())
        print(torch.unique(singleClassMap))

        # [H,W,num_class*num_keypoint*2]
        singleKeyVectorMap = keypointVectorMap[bi]
        
        # Generate hypothesis for each class label
        for c in class_labels:
            in_class_mask = singleClassMap == c
            
            # [pixels_in_class,y,x]
            inClass = in_class_mask.nonzero() 
            inClassCt = inClass.size(0)
            print("Pixesl in class {}: {}".format(c,inClassCt))

            # Skip if no class in segment
            if inClassCt == 0:
                continue


            # Slice to only look at vector output for this class
            # [num_classes*num_keypoints*2, H, W] => [num_keypoints*2, H, W]
            singleClassVectorMap = singleKeyVectorMap[c.item()*2*num_vertices:(c.item()+1)*2*num_vertices,:,:]

            # Apply in_class_mask for keypoint vectors
            vectorPtsInClass = torch.masked_select(
                # To make it easier to repack, permute to [H,W,vectors]
                singleClassVectorMap.permute(1,2,0),
                # Unsqueeze to [H,W,1] for broadcasting
                in_class_mask.unsqueeze(2))

            # Repac masked selection to [pixels_in_class, num_vertices, 2]    
            vectorPtsInClass = vectorPtsInClass.reshape(inClass.size(0),num_vertices,2)            
            
            ###
            ## Run RANSAC
            
            # Termination conditions
            curr_iter = 0
            total_hypotheses = 0
            
            # Holding best answers
            current_vertices = torch.zeros(num_vertices,2,dtype=torch.float)
            current_vertices_votes = torch.zeros(num_vertices,dtype=torch.float)

            while True:
                print("Ransac iteration: {}".format(curr_iter))

                #[num_h,num_v,2]
                hypotheses = _generate_hypotheses(num_hypotheses, vectorPtsInClass, inClass)          
                
                #[num_v, num_h]
                vote_cts = _vote_hypotheses(hypotheses, vectorPtsInClass,inClass,vote_threshold).squeeze(2)
                
                # Per vertex, max across all hypotheses
                max_hypothesis_vote, max_hypothesis_idx = vote_cts.max(1)
                # print(max_hypothesis_idx)

                voted_hypotheses = hypotheses[max_hypothesis_idx,torch.arange(0,num_vertices)]

                # Inlier ratio for each vertex
                max_hypothesis_vote = (max_hypothesis_vote / inClassCt)

                # Update best answers so far 
                better_mask = max_hypothesis_vote > current_vertices_votes
                current_vertices[better_mask] = voted_hypotheses[better_mask]
                current_vertices_votes[better_mask] = max_hypothesis_vote[better_mask]

                # Break if max iterations, or if we exceed confidence threshold
                total_hypotheses += num_hypotheses
                curr_iter += 1

                print("Worst_keypoint score: {}".format(current_vertices_votes.min()))
                if curr_iter > max_iterations or (1 - (1 - current_vertices_votes.min() ** 2) ** total_hypotheses) > confidence:
                    break
            
            return(current_vertices, inClass, hypotheses, vote_cts, vectorPtsInClass) 


"""

Generate num_hypotheses for each keypoint by randomly sampling two points in vectorLookup which are 
assumed to all be within the class for that batch, and computing intersections.

Returns [num_h, num_v, 2], where the final channel specifies [u,v] coords for each hypothesis

"""

def _generate_hypotheses(num_hypotheses,
        vectorPts,
        image_coords
        ):
    num_pts, num_verts, dimw = vectorPts.size()

    # Pick A and B points for num_hypotheses guesses
    A = torch.randint(low=0,high=num_pts-1,size=(num_hypotheses,))
    B = torch.randint(low=0,high=num_pts-1,size=(num_hypotheses,))

    # Given a point picked by random idx A, get the slope at that point, and the [v,u] coordinates of that point 
    A_slope = vectorPts[A]
    A_pts = image_coords[A].unsqueeze(1) # Make broadcastable in num_vertices axis

    # Given a point (u,v) on a line, and a slope (horizontal, vertical), the a,b,c formulation of a line is [vertical, -horizontal, horizontal*v - vertical*u]
    l_a = torch.stack([A_slope[:,:,1],
            -A_slope[:,:,0],
            A_slope[:,:,0]*A_pts[:,:,0] - A_slope[:,:,1]*A_pts[:,:,1]], dim=2)

    B_slope = vectorPts[B]
    B_pts = image_coords[B].unsqueeze(1)
    l_b = torch.stack([B_slope[:,:,1],
            -B_slope[:,:,0],
            B_slope[:,:,0]*B_pts[:,:,0] - B_slope[:,:,1]*B_pts[:,:,1]], dim=2)

    # Intersection of two lines is the cross-product, normalized by 3rd coordinate
    intersections = torch.cross(l_a, l_b,dim=2)
    intersections = intersections / intersections[:,:,2:3]
    intersections = intersections[:,:,0:2] # Drop homogenous coordinate
    


    return(intersections) # [num_hypotheses, num_vertices, 2]



"""

All points in class vote for hypothesis each keypoint by randomly sampling two points in vectorLookup which are 
assumed to all be within the class for that batch, and computing intersections

Returns [num_v, num_h], where value is # of votes for each hypothesis for each vertex

"""

def _vote_hypotheses(
        hypotheses,     #[num_h,num_v,2]
        vectorPts,      #[inclass,num_v,2]
        image_coords,   #[inclass,2]
        vote_threshold):

    broadcastable_hypotheses = hypotheses.permute(1,0,2).unsqueeze(0) #[1,num_v,num_h, 2]

    # Make image_coords broadcastable in num_vertices (dim=1) and num_hypotheses axis (dim2) 
    broadcastable_imagecoords = image_coords.unsqueeze(1)
    broadcastable_imagecoords = broadcastable_imagecoords.unsqueeze(2) #[num_pts, 1, 1, 2]
    broadcastable_imagecoords = broadcastable_imagecoords.flip(3)   # flip x and y

    direction_deltas = broadcastable_hypotheses - broadcastable_imagecoords #[num_pts,num_v,num_h,2]
    direction_deltas = direction_deltas / (direction_deltas.norm(dim=3, p=2).unsqueeze(3))
       
    # Normalize direction predictions to the unit vector, otherwise thresholding won't work
    broadcastable_vectors = vectorPts.unsqueeze(2) #[num_pts, num_v, 1, 2]
    broadcastable_vectors = (broadcastable_vectors / broadcastable_vectors.norm(dim=3, p=2).unsqueeze(3))

    #[num_v,num_h,1]
    return ((direction_deltas @ broadcastable_vectors.permute(0,1,3,2))\
        >= vote_threshold)\
        .sum(dim=0)