from sknano.generators import SWNTBundleGenerator
bundle = SWNTBundleGenerator(n=10, m=5, nz=2, bundle_geometry='hexagon')
bundle.save_data()