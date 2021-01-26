import numpy as np
from .. import dyn_util


def remove_isolated_contacts(contact_dict, threshold=int(2e6)):
    """
    Select only contacts which are within a given sequence separation of another
    contact, for the same chromosome pair
    """

    for chromoA in contact_dict:
        for chromoB in contact_dict[chromoA]:
            contacts = contact_dict[chromoA][chromoB]
            positions = np.array(contacts[:2], np.int32).T
            
            if len(positions): # Sometimes empty e.g. for MT, Y chromos 
                active_idx = dyn_util.getSupportedPairs(positions, np.int32(threshold))
                contact_dict[chromoA][chromoB] = contacts[:,active_idx]

    return contact_dict
    

def remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos, particle_size, threshold=5.0):    
    """
    Remove contacts whith structure distances that exceed a given threshold
    """

    for chr_a in contact_dict:
        for chr_b in contact_dict[chr_a]:
            contacts = contact_dict[chr_a][chr_b]

            contact_pos_a = contacts[0].astype(np.int32)
            contact_pos_b = contacts[1].astype(np.int32)

            coords_a = coords_dict[chr_a]
            coords_b = coords_dict[chr_b]

            struc_dists = []
 
            for m in range(len(coords_a)):
                coord_data_a = dyn_util.getInterpolatedCoords(
                    [chr_a], {chr_a: contact_pos_a},
                    {chr_a: particle_seq_pos[chr_a]},
                    coords_a[m])
                coord_data_b = dyn_util.getInterpolatedCoords(
                    [chr_b], {chr_b: contact_pos_b},
                    {chr_b: particle_seq_pos[chr_b]},
                    coords_b[m])

                deltas = coord_data_a - coord_data_b
                dists = np.sqrt((deltas*deltas).sum(axis=1))
                struc_dists.append(dists)

            # Average over all conformational models
            struc_dists = np.array(struc_dists).T.mean(axis=1)

            # Select contacts with distances below distance threshold
            indices = (struc_dists < threshold).nonzero()[0]
            contact_dict[chr_a][chr_b] = contacts[:,indices]

    return contact_dict
    