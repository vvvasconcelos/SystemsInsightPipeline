import re
import pandas as pd
import numpy as np
import warnings
import networkx as nx
from types import SimpleNamespace
from openpyxl import load_workbook

class Extract:
    def __init__(self, file_path):
        self.file_path = file_path
        self.variables = []
        self.var_to_type = {}
        self.adjacency_matrix = None
        self.interactions_matrix = None

    def extract_settings(self, double_factor_interventions_setting=None):
        """ Extract all settings based on the json and Kumu files
        """
        # Load the adjacency matrix from the KUMU file
        self.adjacency_matrix_from_kumu()

        if np.abs(self.interactions_matrix).sum() > 0:  # Interaction terms specified
            print("Solving an SDM with interaction terms.")
            print("By default, only single interventions will be simulated. To simulate interventions on two variables simultaneously, set s.double_factor_interventions = True")
            s = SimpleNamespace(**{"interaction_terms" : 1,
                                   "solve_analytically" : 0,
                                   "double_factor_interventions" : 0})  # Default to single-factor interventions
        else:
            if getattr(self, "equations", None):
                print("No interaction terms specified; the custom equations are integrated numerically.")
            else:
                print("No interaction terms specified so will solve linear SDM.")
            s = SimpleNamespace(**{"interaction_terms" : 0,
                                   "solve_analytically" : 0,
                                   "double_factor_interventions" : 0})

        if double_factor_interventions_setting is not None: # If specified, override the default setting
            s.double_factor_interventions = double_factor_interventions_setting

        if s.double_factor_interventions and s.interaction_terms == False:
            warnings.warn("Without interaction terms, double factor interventions are not meaningful." \
                          "Consider setting double_factor_interventions to False.")

        s.stocks = [var for var in self.variables if self.var_to_type[var].lower() == 'stock']
        s.auxiliaries = [var for var in self.variables if self.var_to_type[var].lower() == 'auxiliary']
        s.constants = [var for var in self.variables if self.var_to_type[var].lower() == 'constant']
        s.variables = [var for var in self.variables]  # s.auxiliaries + s.stocks + s.constants
        s.stocks_and_constants = [var for var in self.variables if self.var_to_type[var] in ['stock', 'constant']]
        s.stocks_and_auxiliaries = [var for var in self.variables if self.var_to_type[var] in ['stock', 'auxiliary']]
        s.var_to_type = {var : self.var_to_type[var] for var in self.variables}

        s.variable_of_interest = []
        for voi in self.variable_of_interest:
            #voi = voi.replace(" ", "_")
            s.variable_of_interest += [voi]

        s.centrality = self.centrality
        s.intervention_variables = self.intervention_variables #[var.replace(" ", "_") for var in self.intervention_variables]
        s.intervention_strengths = self.intervention_strengths #{x.replace(" ", "_") : self.intervention_strengths[x] for x in self.intervention_strengths}

        if len(s.intervention_variables) == 0:
            raise(Exception("There should be at least one intervention variable specified in the Excel file."))
        
        # If double factor interventions selected, add double factor interventions 
        if s.double_factor_interventions:
            double_intervention_variables = []
            for i, var in enumerate(s.intervention_variables):
                for j in range(i + 1, len(s.intervention_variables)):
                    var_2 = s.intervention_variables[j]
                    double_intervention_variables += [var + '+' + var_2]
            
            s.intervention_variables += double_intervention_variables

        self.df_adj = self.df_adj.rename(index=dict(zip(self.variables, s.variables)),
                                         columns=dict(zip(self.variables, s.variables)))

        s.df_adj = self.df_adj  # Save the adjacency matrix to the settings
        s.interactions_matrix = self.interactions_matrix # Save the interactions matrix to the settings

        # Add the interactions to the adjacency matrix for the identification of feedback loops with interaction terms
        s.df_adj_incl_interactions = s.df_adj.copy()
        to_list, from1_list, from2_list = np.nonzero(s.interactions_matrix)
        for i in range(len(to_list)):  # Use length of nonzero indices, not sum of values
            to, from1, from2 = to_list[i], from1_list[i], from2_list[i]
            value = s.interactions_matrix[to, from1, from2]
            # Ensure that the interaction is nonzero in the adjacency matrix
            s.df_adj_incl_interactions.loc[s.df_adj_incl_interactions.index[to],
                                           s.df_adj_incl_interactions.columns[from1]] = value
            s.df_adj_incl_interactions.loc[s.df_adj_incl_interactions.index[to],
                                           s.df_adj_incl_interactions.columns[from2]] = value
        
        # Add custom equations to settings
        s.equations = self.equations if hasattr(self, 'equations') else {}
        s.equation_warnings = self.equation_warnings if hasattr(self, 'equation_warnings') else []
        # Ranges of named equation parameters (#name) from the optional Parameters sheet
        s.parameter_priors = dict(getattr(self, 'parameter_priors', {}))

        self.s = s  # Save the settings

        return s

    def check_loops(self, df_e, df_c):
        """ Check whether all loops have stocks and the ratio of balancing loops in the CLD
        """
        stocks = [var for var in self.variables if self.var_to_type[var].lower() == "stock"]

        num_stocks_and_auxiliaries = len([var for var in self.variables 
                                        if self.var_to_type[var].lower() in ["stock", "auxiliary"]])
    
        # ## Check for balancing loops
        max_loops_check = 5  # Maximum number of loops to check to not take too much time
        if num_stocks_and_auxiliaries < max_loops_check:
            max_loops_check = num_stocks_and_auxiliaries

        G = nx.from_numpy_array(np.array(self.df_adj).T, create_using=nx.DiGraph)
        var_names = [var for var in self.df_adj.columns]
        G = nx.relabel_nodes(G, dict(enumerate(var_names)))
        # canonical (session-independent) loop order, see ltm.find_feedback_loops
        from .ltm import find_feedback_loops
        feedback_loops = find_feedback_loops(self.df_adj, max_length=max_loops_check, require_stock=False)
        num_loops = len(feedback_loops)

        if num_loops > 0:
            print("\n" + str(num_loops), "feedback loops of maximum length", max_loops_check)

            ### Check if any loops have no stocks
            loops_wo_stocks = []

            for loop in feedback_loops:
                if sum([1 for x in loop if x in stocks]) == 0:
                    loops_wo_stocks += [loop]

            if len(loops_wo_stocks) > 0:
                print(len(loops_wo_stocks), "loops do not have a stock, which is",
                    round((len(loops_wo_stocks)/len(feedback_loops))*100, 5), "% of all loops")
                print("Loops without stocks:", loops_wo_stocks)
                raise(Exception("All loops should have at least one stock, redo the labeling"))
            else:
                print("All loops have at least one stock")
            
            ### Check the ratio of balancing loops
            if (self.df_adj == -999).any().any() > 0:
                num_balancing = 0
                balancing_loops = []
                for loop in feedback_loops:
                    num_min = 0
                    loop += [loop[0]]  # Add the first element to the end to close the loop
                    for i in range(len(loop)-1):
                        pol = df_c.loc[((df_c.From==loop[i])*1 + (df_c.To==loop[i+1])*1) == 2, "Type"].values[0]
                        if str(pol) == "-":
                            num_min += 1
                    if num_min % 2 != 0:  # If the number of negative polarities is odd
                        num_balancing += 1
                        balancing_loops += [loop]

                print(num_balancing, "(" + str(round((num_balancing/num_loops)*100, 2))+
                    "%) of these loops are balancing loops")
                if max_loops_check==num_stocks_and_auxiliaries:
                    print("The max length of loops checked is equal to the number of stocks and auxiliaries; all loops are considered\n")
                else:
                    print("The max length of loops checked is smaller than the number of stocks and auxiliaries; there may be more loops in the CLD\n")
        else:
            print("No feedback loops found in the CLD")

        # Calculate betweenness and closeness centrality
        bc = nx.betweenness_centrality(G, k=None, normalized=True)
        cc = nx.closeness_centrality(G.reverse(), wf_improved=False) # Apply to G.reverse() for outward distance vs inward
        bc = dict(sorted(bc.items(), key=lambda item: item[1], reverse=True))
        cc = dict(sorted(cc.items(), key=lambda item: item[1], reverse=True))

        self.centrality = {"betweenness" : bc, "closeness" : cc}

        
    def extract_adjacency_matrix(self):
        """Extract the adjacency matrix from an Excel table exported from Kumu (Kumu.io).
        The Kumu excel file contains one sheet with the CLD's variables ('Elements').
        It also contains a sheet with the CLD's causal links ('Connections').
        If there are known interactions in the system, these can be added in the 'Interactions' sheet.
        """
        # Read the elements, connections, and interactions sheets in the Kumu Excel file
        df_e = pd.read_excel(self.file_path, sheet_name="Elements")
        df_c = pd.read_excel(self.file_path, sheet_name="Connections")

        # Extract relevant columns - include Equation if present
        base_columns = ["Label", "Type", "Tags", "Description"]
        if "Equation" in df_e.columns:
            base_columns.append("Equation")
            self.has_equations = True
        else:
            self.has_equations = False
        # Tolerate missing annotation columns: create them empty with a clear warning,
        # instead of crashing with a raw KeyError further down.
        if "Label" not in df_e.columns:
            raise Exception("The Elements sheet must have a 'Label' column with the variable names.")
        for col, hint in (("Type", "variables will default to stock/constant based on incoming links"),
                          ("Tags", "no variable will be marked as an intervention"),
                          ("Description", "no variable will be marked as the variable of interest (VOI)")):
            if col not in df_e.columns:
                warnings.warn(f"The Elements sheet has no '{col}' column; {hint}.")
                df_e = df_e.copy()
                df_e[col] = np.nan
        df_e = df_e[[col for col in base_columns if col in df_e.columns]]
        df_c = df_c[["From", "Type", "To"]]
        
        # Extract variables from the Elements
        for row, label in enumerate(df_e["Label"]):
            if pd.isna(label) or str(label).strip() == "":
                raise Exception(f"Row {row + 2} of the Elements sheet has an empty Label.")
        df_e = df_e.copy()
        df_e["Label"] = [str(label) for label in df_e["Label"]]
        self.original_variables = list(df_e["Label"])

        for var in self.original_variables:
            if "+" in var:
                raise(Exception(f'Variable name {var} contains a disallowed special character (+). Please remove these characters from the variable names.'))
            if "*" in var:
                raise(Exception(f'Variable name {var} contains a disallowed special character (*). Please remove these characters from the variable names.'))

        self.variables = [" ".join(var.split()) for var in self.original_variables] # Just remove extra spaces
        # Create dictionary with relevant labels; normalize case so "Stock"/"stock" in the Excel are equivalent
        self.var_to_type = dict(zip(self.variables, [str(t).strip().lower() for t in df_e["Type"]]))
        self.original_to_cleaned_var = dict(zip(self.original_variables, self.variables))

        ### Assign stock or constant to variables without type
        for var in self.original_variables:
            if str(self.var_to_type[self.original_to_cleaned_var[var]]).lower() not in ['stock', 'auxiliary', 'constant']:
                if var in list(df_c['To']): # If it has ingoing links and is therefore endogenous, assign as stock
                    print(f'Warning: Variable {self.original_to_cleaned_var[var]} has no (known) label assigned, namely {self.var_to_type[self.original_to_cleaned_var[var]]}, defaulting to stock as it has ingoing links.')
                    self.var_to_type[self.original_to_cleaned_var[var]] = 'stock'
                else:  # If exogenous, assign as constant
                    print(f'Warning: Variable {self.original_to_cleaned_var[var]} has no (known) label assigned, namely {self.var_to_type[self.original_to_cleaned_var[var]]}, defaulting to constant as it has no ingoing links.')
                    self.var_to_type[self.original_to_cleaned_var[var]] = 'constant'
                print("")

        # Intervention strengths come from the Tags column. Coerce to numeric so that
        # blank cells (NaN) and leftover Kumu tag *text* are treated as "not an
        # intervention" (strength 0) instead of silently producing NaN dynamics or a
        # cryptic numpy crash downstream.
        tags_raw = df_e["Tags"]
        tags = pd.to_numeric(tags_raw, errors="coerce")
        non_numeric = tags.isna() & tags_raw.notna() & (tags_raw.astype(str).str.strip() != "")
        if non_numeric.any():
            bad = ", ".join(f"{lbl!r} (Tags={val!r})"
                            for lbl, val in zip(df_e.loc[non_numeric, "Label"], tags_raw[non_numeric]))
            warnings.warn(
                f"Non-numeric Tags treated as 0 (not an intervention): {bad}. "
                "Use a numeric intervention strength (e.g. 1 or -1) to mark interventions."
            )
        tags = tags.fillna(0.0)

        self.intervention_variables = [self.original_to_cleaned_var[var] for var in list(df_e.loc[tags != 0, "Label"])]
        self.intervention_strengths = dict(zip(self.variables, [float(t) for t in tags]))
        self.variable_of_interest = list(df_e.loc[df_e["Description"] == "VOI", "Label"])

        if len(self.variable_of_interest) == 1:
            print("Variable of interest:", self.variable_of_interest[0])
        else:
            print("Variables of interest:", self.variable_of_interest)
        print("with", len(self.intervention_variables), "intervention variables")

        # Create an empty adjacency matrix
        num_variables = len(self.variables)
        self.adjacency_matrix = np.zeros((num_variables,
                                          num_variables))

        # Populate the adjacency matrix
        for i, origin in enumerate(df_c["From"]):
            if origin not in self.original_variables:
                raise(Exception(f'Origin variable {origin} in Connections sheet not found in Elements sheet. Please check the Kumu Excel file.'))
    
            destination = df_c["To"][i]

            if destination not in self.original_variables:
                raise(Exception(f'Destination variable {destination} in Connections sheet not found in Elements sheet. Please check the Kumu Excel file.'))

            # Determine the polarity
            polarity = 0
            temp = df_c["Type"][i]
            if str(temp) == '+':
                polarity = 1
            elif str(temp) == '-':
                polarity = -1
            else:  # No polarity specified
                polarity = -999

            # Calculate indices
            origin_index = self.original_variables.index(origin)
            destination_index = self.original_variables.index(destination)

            # Add polarity to adjacency matrix
            self.adjacency_matrix[destination_index, origin_index] = polarity

        # Create dataframe with adjacency matrix (an explicit copy, so that the DataFrame and
        # self.adjacency_matrix never share memory, whatever the pandas version)
        self.df_adj = pd.DataFrame(self.adjacency_matrix.copy(),
                                   columns=self.variables,
                                   index=self.variables)

        constants = [var for var in self.variables
                     if self.var_to_type[var].lower() == 'constant']

        ### Constants are exogenous: links into a constant are ignored (with a warning)
        for const in constants:
            sources = [src for src in self.variables if self.df_adj.loc[const, src] != 0]
            if sources:
                warnings.warn(f"Constant '{const}' has {len(sources)} incoming link(s) (from {sources}); "
                              "constants are exogenous, so these links are ignored. Make it a stock or "
                              "an auxiliary if it should depend on other variables.")
                self.df_adj.loc[const, :] = 0
        self.adjacency_matrix = self.df_adj.to_numpy(dtype=float, copy=True)

        # Variables with a custom equation may have no incoming links (e.g. a clock, dC/dt = 1)
        with_equation = set()
        if self.has_equations:
            for label, eq in zip(df_e["Label"], df_e["Equation"]):
                if not (pd.isna(eq) or str(eq).strip() == ""):
                    with_equation.add(self.original_to_cleaned_var[label])

        for var in self.variables:
            if var not in constants and var not in with_equation:
                num_incoming_links = np.sum(np.abs(self.df_adj.loc[var, :]))
                if num_incoming_links == 0:
                    raise(Exception(f'Number of incoming links for (non-constant) variable {var} is 0, '
                                    'should be at least one (or give the variable a custom Equation).'))

        self.check_loops(df_e, df_c)  # Check for stocks in loops and ratio of balancing loops

        # Parse custom equations if present
        self.equations = {}
        self.equation_warnings = []
        if self.has_equations:
            self._parse_equations(df_e)
        self.parameter_priors = self._read_parameter_priors()

    def _parse_equations(self, df_e):
        """
        Parse custom equations from the Equation column.
        
        Validates equations against incoming links and prepares them for evaluation.
        Uses # symbol for parameters (positive range) and supports np.FUNCTION syntax.
        """
        from .equations import EquationParser

        parser = EquationParser(self.variables, var_types=self.var_to_type)

        for idx, row in df_e.iterrows():
            label = row['Label']
            equation = row.get('Equation', None)

            if pd.isna(equation) or str(equation).strip() == '':
                continue

            # Find cleaned variable name
            var_name = self.original_to_cleaned_var.get(label, label)

            if var_name not in self.variables:
                continue

            # Get incoming links from adjacency matrix
            incoming_links = [
                col for col in self.df_adj.columns
                if self.df_adj.loc[var_name, col] != 0
            ]

            # Parse the equation
            parsed = parser.parse_equation(var_name, str(equation), incoming_links)

            if parsed['is_valid']:
                self.equations[var_name] = parsed
                if self.var_to_type.get(var_name) == 'constant':
                    message = (f"Equation of constant '{var_name}' is ignored: constants are exogenous. "
                               "Make it an auxiliary (or a stock) to use the equation.")
                    warnings.warn(message)
                    self.equation_warnings.append(f"[{var_name}] {message}")

                # Print equation info
                print(f"Custom equation for '{var_name}': {equation}")
                print(f"  Variables used: {parsed['variables_used']}")
                print(f"  Parameters (#): {parsed['n_parameters']}")
                if parsed.get('named_parameters'):
                    print(f"  Named parameters: {', '.join('#' + p for p in parsed['named_parameters'])}")
                if parsed.get('has_intervention', False):
                    print(f"  Intervention ($): Yes - intervention applied within equation")
                else:
                    print(f"  Intervention ($): No - intervention will be added to result")
                
                # Collect and print warnings
                for warning in parsed['validation_warnings']:
                    warning_msg = f"  ⚠ {warning}"
                    self.equation_warnings.append(f"[{var_name}] {warning}")
                    print(warning_msg)
                print()
            else:
                # Equation is invalid - print error and raise exception
                print(f"\n❌ INVALID equation for '{var_name}': {equation}")
                for warning in parsed['validation_warnings']:
                    print(f"   {warning}")
                raise ValueError(f"Invalid equation for '{var_name}': {parsed['validation_warnings']}")

    def _read_parameter_priors(self):
        """Read the optional 'Parameters' sheet: one row per named parameter (#name).

        Columns ``Name`` (with or without the leading ``#``), ``Low`` and ``High``. Each named
        parameter is drawn uniformly from ``[Low, High]`` (``Low == High`` fixes it). Named
        parameters that are not listed default to ``[0, parameter_value_aux]``. Numbered
        parameters (``#1``, ``#2``, ``#``) are local to their equation and cannot be listed.
        """
        with pd.ExcelFile(self.file_path) as xl:
            sheets = {str(name).strip().lower(): name for name in xl.sheet_names}
            if "parameters" not in sheets:
                return {}
            df_p = xl.parse(sheets["parameters"])
        cols = {str(c).strip().lower(): c for c in df_p.columns}
        missing = [c for c in ("name", "low", "high") if c not in cols]
        if missing:
            raise ValueError("The Parameters sheet needs the columns Name, Low and High "
                             f"(missing: {', '.join(m.capitalize() for m in missing)}).")
        priors = {}
        for row, record in df_p.iterrows():
            raw = record[cols["name"]]
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            name = str(raw).strip()
            name = name[1:].strip() if name.startswith("#") else name
            where = f"Parameters sheet, row {row + 2}"
            if re.fullmatch(r"\d+", name):
                raise ValueError(f"{where}: '#{name}' is a numbered parameter. Numbered parameters "
                                 "(#1, #2, #) are local to each equation and always range over "
                                 "[0, parameter_value_aux]; give the parameter a name (e.g. #rate) "
                                 "to set its range.")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"{where}: invalid parameter name {raw!r}. Names start with a letter "
                                 "and contain only letters, digits and underscores.")
            low = pd.to_numeric(record[cols["low"]], errors="coerce")
            high = pd.to_numeric(record[cols["high"]], errors="coerce")
            if pd.isna(low) or pd.isna(high):
                raise ValueError(f"{where}: parameter '#{name}' needs numeric Low and High values.")
            if low > high:
                raise ValueError(f"{where}: parameter '#{name}' has Low ({low}) > High ({high}).")
            if name in priors:
                raise ValueError(f"{where}: parameter '#{name}' is listed more than once.")
            priors[name] = (float(low), float(high))

        used = set()
        for parsed in self.equations.values():
            used.update(parsed.get("named_parameters", []))
        unused = sorted(set(priors) - used)
        if unused:
            warnings.warn("Parameters sheet lists parameter(s) not used in any equation: "
                          + ", ".join("#" + p for p in unused))
        if priors:
            print("Named parameter ranges: " + ", ".join(f"#{p} in [{lo:g}, {hi:g}]"
                                                         for p, (lo, hi) in priors.items()))
        return priors

    def extract_interactions_matrix(self):
        """Extract the interactions matrix from the 'Interactions' sheet in the Kumu Excel file."""
        wb = load_workbook(self.file_path, read_only=True)   # open an Excel file and return a workbook
        sheet_names = list(wb.sheetnames)
        wb.close()  # read-only workbooks keep the file open until closed

        # Create an empty matrix to annotate interactions
        num_variables = len(self.variables)
        self.interactions_matrix = np.zeros((num_variables, num_variables, num_variables))

        if 'Interactions' in sheet_names:
            df_i = pd.read_excel(self.file_path, sheet_name="Interactions")
            df_i = df_i[["From1", "From2", "Type", "To"]]

            # Create an empty matrix to annotate interactions
            #num_variables = len(self.variables)
            #self.interactions_matrix = np.zeros((num_variables, num_variables, num_variables))

            # Populate the interactions matrix
            for i, origin_1 in enumerate(df_i["From1"]):
                origin_2 = df_i["From2"][i]
                destination = df_i["To"][i]

                # Determine the polarity
                polarity = 0
                temp = df_i["Type"][i]
                if str(temp) == '+':
                    polarity = 1
                elif str(temp) == '-':
                    polarity = -1
                else:
                    polarity = -999 # Placeholder for missing values

                # Calculate indices
                origin_1_index = self.original_variables.index(origin_1)
                origin_2_index = self.original_variables.index(origin_2)
                destination_index = self.original_variables.index(destination)

                # Add polarity to interactions matrix
                self.interactions_matrix[destination_index, origin_2_index, origin_1_index] = polarity

    def adjacency_matrix_from_kumu(self):
        """Run the CLD analysis by extracting the adjacency matrix and interactions matrix."""
        self.extract_adjacency_matrix()
        self.extract_interactions_matrix()
