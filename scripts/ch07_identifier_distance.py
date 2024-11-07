import opendp.prelude as dp
from ch03_d_sym import d_Sym
import pandas as pd


def id_distance(id_column):
    return dp.user_distance(f"IdDistance({id_column})")


def d_Id(x, x_p, identifiers):
    """compute the identifier distance"""
    return d_Sym(identifiers(x), identifiers(x_p))


if __name__ == "__main__":

    colnames = ["Employee Id", "Date", "Time", "Domain"]
    bl_lines = """872 | 08-01-2020 | 12:00 | mail.com
    867 | 10-01-2020 | 11:00 | games.com
    934 | 11-01-2020 | 08:00 | social.com
    872 | 09-15-2020 | 17:00 | social.com
    867 | 11-13-2020 | 05:00 | mail.com
    014 | 10-27-2020 | 13:00 | social.com""".split("\n")
    
    bl_lines_p = """872 | 08-01-2020 | 12:00 | mail.com
    934 | 11-01-2020 | 08:00 | social.com
    872 | 09-15-2020 | 17:00 | social.com
    014 | 10-27-2020 | 13:00 | social.com""".split("\n")

    print([l.split("|") for l in bl_lines])
    bl = pd.DataFrame(
        [l.split("|") for l in bl_lines],
        columns = colnames)
    
    bl_p = pd.DataFrame(
        [l.split("|") for l in bl_lines_p],
        columns = colnames)

    print(d_Id(bl, bl_p, identifiers=lambda x: set(x["Employee Id"]))) # -> 1
