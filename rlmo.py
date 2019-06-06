from pyscf.lo import pipek
import numpy
from functools import reduce
from pyscf.lo import orth
from pyscf import lib

def atomic_pops(mol, mo_coeff, method='meta_lowdin'):
    '''
    Kwargs:
        method : string
            one of mulliken, lowdin, meta_lowdin

    Returns:
        A 3-index tensor [A,i,j] indicates the population of any orbital-pair
        density |i><j| for each species (atom in this case).  This tensor is
        used to construct the population and gradients etc.
        
        You can customize the PM localization wrt other population metric,
        such as the charge of a site, the charge of a fragment (a group of
        atoms) by overwriting this tensor.  See also the example
        pyscf/examples/loc_orb/40-hubbard_model_PM_localization.py for the PM
        localization of site-based population for hubbard model.
    '''
    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        s = mol.pbc_intor('int1e_ovlp_sph', hermi=1)
    else:
        s = mol.intor_symmetric('int1e_ovlp')
    nmo = mo_coeff.shape[1]
    proj = numpy.empty((mol.natm,nmo,nmo))

    if method.lower() == 'mulliken':
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            csc = reduce(numpy.dot, (mo_coeff[p0:p1].conj().T, s[p0:p1], mo_coeff))
            proj[i] = (csc + csc.conj().T) * .5

    elif method.lower() in ('lowdin', 'meta_lowdin'):
        c = orth.restore_ao_character(mol, 'ANO')
        csc = reduce(lib.dot, (mo_coeff.conj().T, s, orth.orth_ao(mol, method, c, s=s)))
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            proj[i] = numpy.dot(csc[:,p0:p1], csc[:,p0:p1].conj().T)
    else:
        raise KeyError('method = %s' % method)

    units = mol.units
    nunit = len(units)
    region_proj = numpy.zeros((nunit,nmo,nmo))
    
    for i in range(nunit):
        low = units[i][0] - 1
        up  = units[i][1]
        for j in range(low, up):
            region_proj[i] += proj[j]
    
    return region_proj


class RLMO(pipek.PipekMezey):

    def atomic_pops(self, mol, mo_coeff, method=None):
        if method is None:
            method = self.pop_method
        return atomic_pops(mol, mo_coeff, method)


if __name__ == '__main__':
    from pyscf import gto, scf
    
    mol = gto.Mole()
    mol.atom = '''
               H    0.0   0.   0.
               O    0.96   0.   0.
               H    1.200364804   0.929421734   0.
               H    2.5   0.   0.
               O    3.46   0.   0.
               H    3.700364804   0.929421734   0.
               '''
    mol.basis = 'sto3g'
    mol.units = [[1, 3], [4, 6]]

    mf = scf.RHF(mol).run()

    occlim = numpy.where(mf.get_occ() < 0.1)[0][0]
    mo = RLMO(mol).kernel(mf.mo_coeff[:,:occlim], verbose=4)
    moround = numpy.round(mo, 1)
    print(moround)

    
    """
    mol = gto.Mole()
    mol.atom = '''
                C    4.80585    -0.85311    1.96913  
                H    5.1625    -1.86192    1.96913  
                H    5.16252    -0.34871    2.84278  
                H    5.16252    -0.34871    1.09548  
                C    3.26585    -0.85309    1.96913  
                H    2.9092    0.15572    1.97108  
                H    2.90918    -1.35579    1.0945  
                C    2.75251    -1.58148    3.22512  
                H    3.10949    -1.079    4.09975  
                H    3.10885    -2.5904    3.22298  
                C    1.21251    -1.58099    3.22539  
                H    0.85553    -2.0834    2.35072  
                H    0.85616    -0.57208    3.22762  
                H    0.71486    -2.13574    4.355959
               '''
    mol.units = [[1,7], [8, 14]]
    
    mf = scf.RHF(mol).run()

    occlim = numpy.where(mf.get_occ() < 0.1)[0][0]
    mo = RLMO(mol).kernel(mf.mo_coeff[:,:occlim], verbose=4)
    moround = numpy.round(mo, 1)

    print(moround)
    """

    """
    mol = gto.Mole()
    mol.atom = '''
               H    0.0   0.   0.
               O    0.96   0.   0.
               H    1.200364804   0.929421734   0.
               H    2.5   0.   0.
               O    3.46   0.   0.
               H    3.700364804   0.929421734   0.
               '''
    mol.basis = 'sto3g'
    mol.units = [[1, 3], [4, 6]]

    mf = scf.RHF(mol).run()

    print(mf.get_occ())
    mo = RLMO(mol).kernel(mf.mo_coeff[:,:10], verbose=4)
    moround = numpy.round(mo, 1)
    print(moround)
"""