
from Psience.Molecools import Molecule
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem


embedding_properties_name_map = {
    'use_experimental_torsion_angles':'useExpTorsionAnglePrefs',
    'use_basic_knowledge':'useBasicKnowledge',
}
def get_embedding_options(
        *,
        use_experimental_torsion_angles=True,
        use_basic_knowledge=True,
        **embedding_options
):
    params = AllChem.ETKDGv3()
    cust_opts = dict(
        use_basic_knowledge=use_basic_knowledge,
        use_experimental_torsion_angles=use_experimental_torsion_angles
    )
    for k,v in cust_opts.items():
        mapped = embedding_properties_name_map[k]
        embedding_options[mapped] = embedding_options.get(mapped, v)
    for k,v in embedding_options.items():
        setattr(params, k, v)
    return params

def generate_rdkit_conformer_set(mol:Chem.Mol, num_confs, **embedding_options):
    conf_set = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        params=get_embedding_options(**embedding_options)
    )
    return conf_set

def convert_conformer_set(mol:Chem.Mol, conf_ids):
    return [
        Molecule.from_rdmol(mol.GetConformer(conf_id))
        for conf_id in conf_ids
    ]


def generate_rdkit_conformers(mol:'Chem.Mol|Molecule', num_confs, **embedding_options):
    if isinstance(mol, Molecule):
        mol = mol.rdmol.rdmol

    return convert_conformer_set(
        mol,
        generate_rdkit_conformer_set(mol, num_confs, **embedding_options)
    )
