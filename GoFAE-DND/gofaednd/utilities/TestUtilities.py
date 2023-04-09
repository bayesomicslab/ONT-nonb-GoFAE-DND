import sys
from hypotests.CoreTests import AndersonDarling, CramerVonMises, KolmogorovSmirnov, \
    ShapiroFrancia, ShapiroWilk, HenzeZirkler, MardiaSkew, Royston, EppsPulley1


def testsetup(test, emppath, test_dictionary):
    if test == 'hz':
        return HenzeZirkler(emppath, test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["n"],
                            test_dictionary["n_z"], test_dictionary["use_emp"])
    elif test == 'sw':
        return ShapiroWilk(emppath, test_dictionary["sw_wts"], test_dictionary["sw_mu"],
                           test_dictionary["sw_sigma"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
    elif test == 'sf':
        return ShapiroFrancia(emppath, test_dictionary["sf_wts"], test_dictionary["sf_mu"], test_dictionary["sf_sigma"],
                              test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
    elif test == 'ad':
        return AndersonDarling(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
    elif test == 'cvm':
        return CramerVonMises(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
    elif test == 'ep1':
        return EppsPulley1(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
    elif test == 'ks':
        return KolmogorovSmirnov(emppath, test_dictionary["n"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
    elif test == 'mardia_skew':
        return MardiaSkew(emppath, test_dictionary["n_z"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
    elif test == 'royston':
        sw=ShapiroWilk(emppath, test_dictionary["sw_wts"], test_dictionary["sw_mu"],
                    test_dictionary["sw_sigma"], test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
        sf=ShapiroFrancia(emppath, test_dictionary["sf_mu"], test_dictionary["sf_sigma"],
                       test_dictionary["device"], test_dictionary["new_stat"], test_dictionary["use_emp"])
        return Royston(emppath, test_dictionary["e"], test_dictionary["device"], sf,
                       sw, test_dictionary["new_stat"], test_dictionary["n"], test_dictionary["use_emp"])
    else:
        sys.exit("Test not implemented")

def check_test(test, latent_dim_proj):
    if not test.is_univariate:
        if latent_dim_proj == 1:
            sys.exit("Project to higher dimension or select a univariate test")
        else:
            pass
    else: # multivariate
        if latent_dim_proj > 1:
            sys.exit("Project to univariate or select a multivariate test")
        else:
            pass


