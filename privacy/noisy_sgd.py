
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition 
from autodp import mechanism_zoo, transformer_zoo

class NoisySGD_mech(Mechanism):
    def __init__(self,prob,sigma,niter,name='NoisySGD'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'prob':prob,'sigma':sigma,'niter':niter}
        
        # create such a mechanism as in previously
        subsample = transformer_zoo.AmplificationBySampling() # by default this is using poisson sampling
        mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
        prob = prob
        # Create subsampled Gaussian mechanism
        SubsampledGaussian_mech = subsample(mech,prob,improved_bound_flag=True)

        # Now run this for niter iterations
        compose = transformer_zoo.Composition()
        mech = compose([SubsampledGaussian_mech],[niter])

        # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
        rdp_total = mech.RenyiDP
        self.propagate_updates(rdp_total,type_of_update='RDP')

        
gamma = 0.01
        
noisysgd = NoisySGD_mech(prob=gamma,sigma=5.0,niter=1000)


# compute epsilon, as a function of delta
noisysgd.get_approxDP(delta=1e-6)