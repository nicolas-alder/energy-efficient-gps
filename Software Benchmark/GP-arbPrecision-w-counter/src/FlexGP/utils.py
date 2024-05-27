import operator
from . import count_enabler
from . import config
from . import cg_tracker
from .load_libraries import *


global current_global_section, global_section_info, wrapped_functions
current_global_section = "all_sections"
global_section_info = {"all_sections": {"precision": None, "emax": None, "emin": None, "exponent": None}}
wrapped_functions = []
wrapped_functions_names = []

def load_config():
    configs = []
    for precision in [3,4,5,6,7,8,9,14,24,34,44,53,64]:
        config = {}
        config["precision"] = {}
        config["emin"] = {}
        config["emax"] = {}

        #precision/mantissa
        config["precision"]["kernel"] = precision
        config["precision"]["section_rbf_kernel_XX"] = precision
        config["precision"]["section_rbf_kernel_X*X"] = precision
        config["precision"]["section_rbf_kernel_XX*"] = precision
        config["precision"]["section_rbf_kernel_X*X*"] = precision
        config["precision"]["section_pivoted_cholesky"] = precision
        config["precision"]["section_conjugate_gradient_mean"] = precision
        config["precision"]["section_conjugate_gradient_covariance"] = precision
        config["precision"]["other"] = precision
        config["precision"]["other_predict"] = precision
        config["precision"]["section_woodbury"] = precision
        config["precision"]["section_cg_mvm_only_mean"] = precision
        config["precision"]["section_cg_mvm_only_covariance"] = precision
        config["precision"]["section_cholesky_mean"] = precision
        config["precision"]["section_cholesky_covariance"] = precision

        #exponents
        for emin,emax in [(-1022, 1023)]:
            config["emin"]["section_rbf_kernel_XX"] = emin
            config["emax"]["section_rbf_kernel_XX"]= emax

            config["emin"]["section_rbf_kernel_X*X"] = emin
            config["emax"]["section_rbf_kernel_X*X"] = emax

            config["emin"]["section_rbf_kernel_XX*"]= emin
            config["emax"]["section_rbf_kernel_XX*"] = emax

            config["emin"]["section_rbf_kernel_X*X*"] = emin
            config["emax"]["section_rbf_kernel_X*X*"] = emax

            config["emin"]["section_pivoted_cholesky"] = emin
            config["emax"]["section_pivoted_cholesky"] = emax

            config["emin"]["section_conjugate_gradient_mean"] = emin
            config["emax"]["section_conjugate_gradient_mean"] = emax

            config["emin"]["section_conjugate_gradient_covariance"]= emin
            config["emax"]["section_conjugate_gradient_covariance"] = emax

            config["emin"]["other"] = emin
            config["emax"]["other"] = emax

            config["emin"]["other_predict"]= emin
            config["emax"]["other_predict"] = emax

            config["emin"]["section_woodbury"] = emin
            config["emax"]["section_woodbury"] = emax

            config["emin"]["section_cg_mvm_only_mean"] = emin
            config["emax"]["section_cg_mvm_only_mean"] = emax

            config["emin"]["section_cg_mvm_only_covariance"] = emin
            config["emax"]["section_cg_mvm_only_covariance"] = emax

            config["emin"]["section_cholesky_mean"] = emin
            config["emax"]["section_cholesky_mean"] = emax

            config["emin"]["section_cholesky_covariance"] = emin
            config["emax"]["section_cholesky_covariance"] = emax

            for method in ["cholesky", "cg"]:

                config["method"] = method

                for dataset_size in [500]:
                    config["dataset_size"] = dataset_size

                    for dataset_type in ["california_housing", "201_pol", "294_satellite_image", "344_mv", "503_wind", "564_fried"]:
                        config["dataset_type"] = dataset_type

                        for length_scale in [1.0]:
                            for repetition in range(5):
                                config["kernel"] = {}
                                config["kernel"]["length_scales"] = np.full(7, gmpy2.mpfr(length_scale))
                                print(length_scale)
                                print(gmpy2.mpfr(length_scale))
                                e_bits = math.ceil(math.log2(abs(emax - emin)))

                                config["name"] = method + "_" + str(dataset_size) + "_" + str(precision) + "_" + str(e_bits) + "_" + str(length_scale) + "_" + dataset_type + "_" + str(repetition)
                                print(config)
                                configs.append(config.copy())

    return configs

def  load_config_files():
    experiments = load_config()

    return experiments

# Convert float to MpMath object
convert2mpf = lambda x: gmpy2.mpfr(x)
converter2mpf = np.vectorize(convert2mpf)

def set_global_section(section):
    if not count_enabler.counting_enabled.condition:
        return
    count_enabler.counting_enabled.old_state = copy.copy(
        count_enabler.counting_enabled.condition)
    count_enabler.counting_enabled.condition = False
    global current_global_section
    global global_section_info
    current_global_section = section
    gmpy2.get_context().precision = config.experiment.config["precision"][section]
    gmpy2.get_context().emin = config.experiment.config["emin"][section]
    gmpy2.get_context().emax = config.experiment.config["emax"][section]
    gmpy2.get_context().subnormalize = True

    if section not in global_section_info:
        global_section_info[section] = {"precision": None, "emax": None, "emin": None, "exponent": None}
        bits_exponent = math.ceil(math.log2(operator.abs(gmpy2.get_context().emax - gmpy2.get_context().emin)))
        global_section_info[section] = {"precision": gmpy2.get_context().precision, "emax": gmpy2.get_context().emax,
                                    "emin": gmpy2.get_context().emin, "exponent": bits_exponent}

    count_enabler.counting_enabled.condition = count_enabler.counting_enabled.old_state

def get_global_section():
    global current_global_section
    return current_global_section

def max_array_len(args):
    max_length = 0
    for arg in args:
        if isinstance(arg, list):
            max_length = max(max_length, len(arg))
        else:
            max_length = max(max_length, 1)
    return max_length
def count(operation):

    def wrapper(*args, **kwargs):

        result = operation(*args, **kwargs)

        global wrapped_functions_names
        global wrapped_functions
        global current_global_section
        global global_section_info


        if wrapper.__name__ not in wrapped_functions_names:
            wrapper.__name__ = operation.__name__
            wrapper.section_counts = {
                "all_sections": {"count": 0, "precision": None, "emax": None, "emin": None, "exponent": None}}
            wrapped_functions.append(wrapper)
            wrapped_functions_names.append(wrapper.__name__)


        section = current_global_section

        if section not in wrapper.section_counts:
            wrapper.section_counts[section] = {
                "count": 0,
                "precision": global_section_info[section]["precision"],
                "emax": global_section_info[section]["emax"],
                "emin": global_section_info[section]["emin"],
                "exponent": global_section_info[section]["exponent"],
            }


        if count_enabler.counting_enabled.condition:

            count_enabler.counting_enabled.condition = False

            if wrapper.__name__ == "len":
                wrapper.section_counts[section]["count"] += 1
            elif wrapper.__name__ == "power_two":
                wrapper.section_counts[section]["count"] += result.size
            elif wrapper.__name__ == "exponential":
                wrapper.section_counts[section]["count"] += result.size
            elif wrapper.__name__ == "absolute":
                wrapper.section_counts[section]["count"] += result.size
            elif wrapper.__name__ == "log":
                wrapper.section_counts[section]["count"] += result.size
            elif wrapper.__name__ == "np.max":
                wrapper.section_counts[section]["count"] += args[0].size-1
            elif wrapper.__name__ == "np.sum":
                wrapper.section_counts[section]["count"] += max(args[0].size-1, 0)
            elif wrapper.__name__ == "range":
                wrapper.section_counts[section]["count"] += len(result)
            elif wrapper.__name__ == "abs" and type(*args) != int and not isinstance(*args, gmpy2.mpfr) :
                wrapper.section_counts[section]["count"] += len(*args)
            elif wrapper.__name__ == "np.add":
                wrapper.section_counts[section]["count"] += result.size
            elif wrapper.__name__ == "np.subtract":
                try:
                    wrapper.section_counts[section]["count"] += result.size
                except:
                    wrapper.section_counts[section]["count"] += 1 # scalar
            elif wrapper.__name__ == "np.divide":
                try:
                    wrapper.section_counts[section]["count"] += result.size
                except:
                    wrapper.section_counts[section]["count"] += 1 # scalar
            elif wrapper.__name__ == "np.multiply":
                wrapper.section_counts[section]["count"] += result.size
            elif wrapper.__name__ == "np.sign":
                wrapper.section_counts[section]["count"] += result.size
            elif wrapper.__name__ == "sqrt_float":
                wrapper.section_counts[section]["count"] += 1
            elif wrapper.__name__ == "np.inner":
                wrapper.section_counts[section]["count"] += args[0].size * 2 - 1
            elif wrapper.__name__ == "np.dot":
                n = args[0].shape[0]
                if args[1].ndim > 1:
                    m, k = args[1].shape
                else:
                    m = args[1].shape[0]
                    k = 1

                add_to_wrapper_counter("np.add", section, n * k * (m - 1))
                add_to_wrapper_counter("np.multiply", section, n * k * m)

            else:
                wrapper.section_counts[section]["count"] += 1

            if section != "all_sections":
                if wrapper.__name__ == "len":
                    wrapper.section_counts["all_sections"]["count"] += 1
                elif wrapper.__name__ == "power_two":
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "exponential":
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "absolute":
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "log":
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "np.max":
                    wrapper.section_counts["all_sections"]["count"] += args[0].size-1
                elif wrapper.__name__ == "np.sum":
                    wrapper.section_counts["all_sections"]["count"] += max(args[0].size-1, 0)
                elif wrapper.__name__ == "range":
                    wrapper.section_counts["all_sections"]["count"] += len(result)
                elif wrapper.__name__ == "abs" and type(*args) != int and not isinstance(*args, gmpy2.mpfr) :
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "np.add":
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "np.subtract":
                    try:
                        wrapper.section_counts["all_sections"]["count"] += result.size
                    except:
                        wrapper.section_counts["all_sections"]["count"] += 1  # scalar
                elif wrapper.__name__ == "np.divide":
                    try:
                        wrapper.section_counts["all_sections"]["count"] += result.size
                    except:
                        wrapper.section_counts["all_sections"]["count"] += 1  # scalar
                elif wrapper.__name__ == "np.multiply":
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "np.sign":
                    wrapper.section_counts["all_sections"]["count"] += result.size
                elif wrapper.__name__ == "sqrt_float":
                    wrapper.section_counts["all_sections"]["count"] += 1
                elif wrapper.__name__ == "np.dot" or wrapper.__name__ == "np.inner":
                    n = args[0].shape[0]
                    if args[1].ndim > 1:
                        m, k = args[1].shape
                    else:
                        m = args[1].shape[0]
                        k = 1
                    add_to_wrapper_counter("np.add", "all_sections", n*k*(m-1))
                    add_to_wrapper_counter("np.multiply", "all_sections",  n*k*m)
                else:
                    wrapper.section_counts["all_sections"]["count"] += 1
            count_enabler.counting_enabled.condition = True # Activate to continue counting
        return result

    wrapper.__name__ = operation.__name__
    wrapper.section_counts = {
        "all_sections": {"count": 0, "precision": None, "emax": None, "emin": None, "exponent": None}}
    wrapped_functions.append(wrapper)
    wrapped_functions_names.append(wrapper.__name__)

    return wrapper

def add_to_wrapper_counter(name, section, count):
    global wrapped_functions
    for wrapped_function in wrapped_functions:
        if wrapped_function.__name__ == name:

            if section not in wrapped_function.section_counts:
                wrapped_function.section_counts[section] = {
                    "count": 0,
                    "precision": global_section_info[section]["precision"],
                    "emax": global_section_info[section]["emax"],
                    "emin": global_section_info[section]["emin"],
                    "exponent": global_section_info[section]["exponent"],
                }

            wrapped_function.section_counts[section]['count'] += count
            return

def reset_wrapped_functions():
    global global_section_info, current_global_section, wrapped_functions, wrapped_functions_names
    global_section_info = {"all_sections": {"precision": None, "emax": None, "emin": None, "exponent": None}}
    current_global_section = "all_sections"
    wrapped_functions = []
    wrapped_functions_names = []

def get_section_count(wrapped_function, section="all_sections"):
    return wrapped_function.section_counts.get(section, {"count": 0})["count"]

def list_all_sections_with_counts_and_info():
    global wrapped_functions
    result = {}
    for wrapped_function in wrapped_functions:
        function_name = wrapped_function.__name__
        result[function_name] = {section: {"count": info["count"], "precision": info["precision"], "emax": info["emax"], "emin": info["emin"], "exponent": info["exponent"]}
                                 for section, info in wrapped_function.section_counts.items()}
    return result

def counts_info_to_csv(output_file):
    info = list_all_sections_with_counts_and_info()
    rows = []

    # Füge den Header hinzu
    header = ["Function", "Section", "Count", "Precision", "EMAX", "EMIN", "Exponent"]
    # rows.append(header)

    total_number_of_counts = 0
    for function_name, sections in info.items():
        row = []
        for section, section_info in sections.items():

            function_name = f"{function_name}"
            count = section_info["count"]
            precision = section_info["precision"]
            emax = section_info["emax"]
            emin = section_info["emin"]
            exponent = section_info["exponent"]

            precision_str = f"{precision}" if precision is not None else "None"
            emax_str = f"{emax}" if emax is not None else "None"
            emin_str = f"{emin}" if emin is not None else "None"
            exponent_str = f"{exponent}" if exponent is not None else "None"

            row.append([function_name, section, count, precision_str, emax_str, emin_str, exponent_str])

            if section == "all_sections":
                total_number_of_counts = total_number_of_counts + count

        rows = rows + row

    output_folder = "experiments/results/" + output_file + "/"
    os.makedirs(output_folder, exist_ok=True)

    with open(output_folder + output_file + "_precision_and_counts_total.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerow(["all_functions", "all_sections", total_number_of_counts, "", "", "", ""])
        for row in rows:
            csv_writer.writerow(row)


def pretty_print_sections_info_to_csv(output_file, condition_number, dataset_size,  rms_flexGP_train, rms_flexGP_test, rms_scikit_train, rms_scikit_test, metric_covDiff_flexGP_train, metric_covDiff_flexGP_test, metric_covDiff_scikit_train, metric_covDiff_scikit_test, covDiff_flexGP_train, covDiff_scikit_train,  covDiff_scikit_test, covDiff_flexGP_test):
    output_folder = "experiments/results/" + output_file + "/"
    os.makedirs(output_folder, exist_ok=True)
    # iteration
    with open(output_folder + output_file + "_cg.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in cg_tracker.tracker.saver:
            csv_writer.writerow(row)

    with open(output_folder + output_file + "_performance.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["condition_number", "dataset_size", "rms_flexGP_train", "rms_flexGP_test", "rms_scikit_train", "rms_scikit_test", "metric_covDiff_flexGP_train", "metric_covDiff_flexGP_test", "metric_covDiff_scikit_train", "metric_covDiff_scikit_test"])
        csv_writer.writerow([condition_number, dataset_size, rms_flexGP_train, rms_flexGP_test, rms_scikit_train, rms_scikit_test, metric_covDiff_flexGP_train, metric_covDiff_flexGP_test, metric_covDiff_scikit_train, metric_covDiff_scikit_test])

    with open(output_folder + output_file + "_covdiff.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["confidence_level", "covDiff_flexGP_train", "covDiff_scikit_train",  "covDiff_scikit_test", "covDiff_flexGP_test"])
        for row, i in enumerate(np.arange(0.0,1.0,0.01)):
            csv_writer.writerow([i, covDiff_flexGP_train[row], covDiff_scikit_train[row],  covDiff_scikit_test[row], covDiff_flexGP_test[row]])

len = count(len)
range = count(range)
lt = count(operator.lt)
eq = count(operator.eq)
gt = count(operator.gt)

def add_int(a, b):
    return operator.add(a, b)

add_int.__name__ = "add_integer"
add_int = count(add_int)

def add_int_untracked(a, b):
    return operator.add(a, b)

add_int_untracked.__name__ = "add_untracked"
add_int_untracked = count(add_int_untracked)

def sub_int(a, b):
    return operator.sub(a, b)

sub_int.__name__ = "sub_integer"
sub_int = count(sub_int)

def sub_int_untracked(a, b):
    return operator.sub(a, b)

sub_int_untracked.__name__ = "sub_untracked"
sub_int_untracked = count(sub_int_untracked)

def mul_int(a, b):
    return operator.mul(a, b)

mul_int.__name__ = "mul_integer"
mul_int = count(mul_int)

def mul_int_untracked(a, b):
    return operator.mul(a, b)

mul_int_untracked.__name__ = "mul_untracked"
mul_int_untracked = count(mul_int_untracked)

def truediv_int(a, b):
    return operator.truediv(a, b)

truediv_int.__name__ = "truediv_integer"
truediv_int = count(truediv_int)

def truediv_int_untracked(a, b):
    return operator.truediv(a, b)

truediv_int_untracked.__name__ = "truediv_untracked"
truediv_int_untracked = count(truediv_int_untracked)

def floordiv_int(a, b):
    return operator.floordiv(a, b)

floordiv_int.__name__ = "floordiv_integer"
floordiv_int = count(floordiv_int)

def floordiv_int_untracked(a, b):
    return operator.floordiv(a, b)

floordiv_int_untracked.__name__ = "floordiv_untracked"
floordiv_int_untracked = count(floordiv_int_untracked)

##############################################################################
# Purpose: Track GMPY2
# Source: Own
##############################################################################

def np_add(x, y):
    return np.add(x, y)

np_add.__name__ = "np.add"
np_add = count(np_add)

def np_subtract(x, y):
    return np.subtract(x, y)

np_subtract.__name__ = "np.subtract"
np_subtract = count(np_subtract)

def np_divide(x, y):
    return np.divide(x, y)

np_divide.__name__ = "np.divide"
np_divide = count(np_divide)

def np_multiply(x, y):
    return np.multiply(x, y)
np_multiply.__name__ = "np.multiply"
np_multiply = count(np_multiply)

def np_dot(x, y):
    return np.dot(x, y)
np_dot.__name__ = "np.dot"
np_dot = count(np_dot)

def np_sum(x):
    return np.sum(x)

np_sum.__name__ = "np.sum"
np_sum = count(np_sum)

def np_inner(x, y):
    return np.inner(x, y)
np_inner.__name__ = "np.inner"
np_inner = count(np_inner)

def np_sign(x):
    return np.sign(x)

np_sign.__name__ = "np.sign"
np_sign = count(np_sign)

def np_max(x):
    return np.max(x)

np_max.__name__ = "np.max"
np_max = count(np_max)


log_gmpy = lambda x: gmpy2.log(x)
log = np.vectorize(log_gmpy) # wird einmal zuviel gezählt durch vectorize
log.__name__ = "log"
log = count(log)

power_gmpy = lambda x: math.pow(x, 2)
power_two = np.vectorize(power_gmpy)
power_two .__name__ = "power_two"
power_two = count(power_two)

exponential_gmpy = lambda x: math.exp(x)
exponential = np.vectorize(exponential_gmpy)
exponential.__name__ = "exponential"
exponential = count(exponential)

abs_gmpy = lambda x: abs(x)
absolute = np.vectorize(abs_gmpy)
absolute.__name__ = "absolute"
absolute = count(absolute)

def add_float(a, b):
    return gmpy2.add(a, b)

add_float = count(add_float)
add_float.__name__ = "add_float"

def mul_float(a, b):
    return gmpy2.mul(a, b)

mul_float.__name__ = "mul_float"
mul_float = count(mul_float)

def sub_float(a, b):
    return gmpy2.sub(a, b)

sub_float.__name__ = "sub_float"
sub_float = count(sub_float)

def truediv_float(a, b):
    return gmpy2.div(a, b)

truediv_float.__name__ = "truediv_float"
truediv_float = count(truediv_float)

def sqrt_float(a):
    return gmpy2.sqrt(a)

sqrt_float.__name__ = "sqrt_float"
sqrt_float = count(sqrt_float)


