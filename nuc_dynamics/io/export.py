import os.path as osp
from ..util.const import N3D, PDB

def export_n3d_coords(file_path, coords_dict, seq_pos_dict, *args, **kwargs):

    file_obj = open(file_path, 'w')
    write = file_obj.write
    
    for chromo in seq_pos_dict:
        chromo_coords = coords_dict[chromo]
        chromo_seq_pos = seq_pos_dict[chromo]
        
        num_models = len(chromo_coords)
        num_coords = len(chromo_seq_pos)
        
        line = '%s\t%d\t%d\n' % (chromo, num_coords, num_models)
        write(line)
        
        for j in range(num_coords):
            data = chromo_coords[:,j].ravel().tolist()
            data = '\t'.join('%.8f' % d for d in    data)
            
            line = '%d\t%s\n' % (chromo_seq_pos[j], data)
            write(line)

    file_obj.close()


def export_pdb_coords(file_path, coords_dict, seq_pos_dict, particle_size, scale=1.0, extended=True):
    """
    Write chromosome particle coordinates as a PDB format file
    """

    alc = ' '
    ins = ' '
    prefix = 'HETATM'
    line_format = '%-80.80s\n'
    
    if extended:
        pdb_format = '%-6.6s%5.1d %4.4s%s%3.3s %s%4.1d%s     %8.3f%8.3f%8.3f%6.2f%6.2f                    %2.2s    %10d\n'
        ter_format = '%-6.6s%5.1d            %s %s%4.1d%s                                                                                                         %10d\n'
    else:
        pdb_format = '%-6.6s%5.1d %4.4s%s%3.3s %s%4.1d%s     %8.3f%8.3f%8.3f%6.2f%6.2f                    %2.2s    \n'
        ter_format = '%-6.6s%5.1d            %s %s%4.1d%s                                                                                                         \n'

    file_obj = open(file_path, 'w')
    write = file_obj.write
    
    chromosomes = list(seq_pos_dict.keys())
    
    sort_chromos = []
    for chromo in chromosomes:
        if chromo[:3].lower() == 'chr':
            key = chromo[3:]
        else:
            key = chromo
        
        if key.isdigit():
            key = '%03d' % int(key)
        
        sort_chromos.append((key, chromo))
    
    sort_chromos.sort()
    sort_chromos = [x[1] for x in sort_chromos]
    
    num_models = len(coords_dict[chromosomes[0]])
    title = 'NucDynamics genome structure export'
    
    write(line_format % 'TITLE         %s' % title)
    write(line_format % 'REMARK 210') 
    write(line_format % 'REMARK 210 Atom type C is used for all particles')
    write(line_format % 'REMARK 210 Atom number increases every %s bases' % particle_size)
    write(line_format % 'REMARK 210 Residue code indicates chromosome')
    write(line_format % 'REMARK 210 Residue number represents which sequence Mb the atom is in')
    write(line_format % 'REMARK 210 Chain letter is different every chromosome, where Chr1=a, Chr2=b etc.')
    
    if extended:
        file_obj.write(line_format % 'REMARK 210 Extended PDB format with particle seq. pos. in last column')
    
    file_obj.write(line_format % 'REMARK 210')
    
    pos_chromo = {}
    
    for m in range(num_models):
        line = 'MODEL         %4d' % (m+1)
        write(line_format    % line)
        
        c = 0
        j = 1
        seqPrev = None
        
        for k, chromo in enumerate(sort_chromos):
            chain_code = chr(ord('A')+k)                        
            
            tlc = chromo
            if tlc.lower().startswith("chr"):
                tlc = tlc[3:]
            while len(tlc) < 2:
                tlc = '_' + tlc
            
            if len(tlc) == 2:
                tlc = 'C' + tlc
            
            if len(tlc) > 3:
                tlc = tlc[:3]
            
            chromo_model_coords = coords_dict[chromo][m]
            
            if not len(chromo_model_coords):
                continue
            
            pos = seq_pos_dict[chromo]
            
            for i, seqPos in enumerate(pos):
                c += 1
 
                seqMb = int(seqPos//1e6) + 1
                
                if seqMb == seqPrev:
                    j += 1
                else:
                    j = 1
                
                el = 'C'
                a = 'C%d' % j
                    
                aName = '%-3s' % a
                x, y, z = chromo_model_coords[i] #XYZ coordinates
                 
                seqPrev = seqMb
                pos_chromo[c] = chromo
                
                if extended:
                    line = pdb_format % (prefix,c,aName,alc,tlc,chain_code,seqMb,ins,x,y,z,0.0,0.0,el,seqPos)
                else:
                    line = pdb_format % (prefix,c,aName,alc,tlc,chain_code,seqMb,ins,x,y,z,0.0,0.0,el)
                    
                write(line)
 
        write(line_format    % 'ENDMDL')
 
    for i in range(c-2):
        if pos_chromo[i+1] == pos_chromo[i+2]:
            line = 'CONECT%5.1d%5.1d' % (i+1, i+2)
            write(line_format    % line)
 
    write(line_format % 'END')
    file_obj.close()


def split_paraticles(coords_dict, particle_seq_pos):
    chromosomes = list(coords_dict.keys())
    for chr_ in chromosomes:
        yield chr_, {chr_: coords_dict[chr_]}, {chr_: particle_seq_pos[chr_]}


def export_coords(
        out_format,
        out_file_path, coords_dict,
        particle_seq_pos, particle_size,
        split_by_chromo=False):

    # Save final coords as N3D or PDB format file
    if not out_file_path.endswith(out_format):
        out_file_path = '%s.%s' % (out_file_path, out_format)
    export_func = export_pdb_coords if out_format == PDB else export_n3d_coords
    if split_by_chromo:
        for chr_, coords, seq_pos in split_paraticles(coords_dict, particle_seq_pos):
            prefix, ext = osp.splitext(out_file_path)
            outpath = f"{prefix}.{chr_}{ext}"
            export_func(outpath, coords, seq_pos, particle_size)
    else:
        export_func(out_file_path, coords_dict, particle_seq_pos, particle_size)
        print('Saved structure file to: %s' % out_file_path)
