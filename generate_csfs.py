import numpy as np
from xml.dom import minidom
import itertools
from spineigenfunctions import getCSF
import sys

def process_xml_csf(finp):
   """
   Process an XML file and returns CSF determinants.
   Parameters:
       finp (str): Path to the XML input file.
   Returns:
       results (list): List of CSFs
   """
   results = []
   doc = minidom.parse(finp)

   # General info
   general = doc.getElementsByTagName("general")
   elec = general[0].getElementsByTagName('electron')
   spin = int(general[0].getElementsByTagName('spin')[0].childNodes[0].data)
   na = elec[0].getAttribute("na")
   nb = elec[0].getAttribute("nb")

   # Get all the blocks
   blocks = doc.getElementsByTagName("block")

   for block in blocks:
       spaces = block.getElementsByTagName('space')
       seqs = []
       occspace = []

       for space in spaces:
           ne = int(space.getAttribute("ne"))
           ip = int(space.getAttribute("imo"))
           fp = int(space.getAttribute("fmo"))

           seq = np.arange(ip, fp + 1)
           seqtot = np.concatenate((seq, seq), axis=0)
           seqs = np.concatenate((seqs, seq), axis=0)

           s = []
           for l in itertools.combinations(seqtot, ne):
               count = [l.count(i) for i in range(ip, fp + 1)]
               s.append(count)

           # Remove duplicates
           occ = []
           for i in s:
               if i not in occ:
                   occ.append(i)

           occspace.append(occ)

       if len(occspace) == 1:
           seqocc = occspace[0]

       for i in range(len(occspace) - 1):
           if i == 0:
               seqocc = occspace[i]
           s = []
           for j in seqocc:
               for k in occspace[i + 1]:
                   s.append(j + k)
           seqocc = s

       for occ in seqocc:
           p = []
           unp = []
           for i in range(len(occ)):
               if occ[i] == 2:
                   p.append(int(seqs[i]))
               elif occ[i] == 1:
                   unp.append(int(seqs[i]))

           if spin == 3 and unp == []:
               continue
           csfs = getCSF(spin, p, unp)
           for csf in csfs:
            results.append(csf)

   return results

if __name__ == "__main__":
 # Example test case
     csfs = process_xml_csf('he.xml')
     ncsfs = len(csfs)
     for csf in csfs:
        print(csf)

