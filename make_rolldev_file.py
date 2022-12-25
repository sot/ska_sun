# For the pitch/roll table.  I used google sheets to save the xlsx file as
# a CSV (after adding headers: pitch, negrolldev, rolldev).
# For that CSV file, this little script checked for matching neg and
# positive roll values and then saved out a compressed file.

# For future versions of the file, we should convert directly
# with something like
# df = pandas.read_excel('pitch_roll_2022.xlsx'
#    ...: , names=['pitch', 'rolldev_minus', 'rolldev_plus'])
# dat = Table.from_pandas(df)

import numpy as np
from astropy.table import Table

dat = Table.read("pitch_roll_2022.csv")

assert np.allclose(np.abs(dat["negrolldev"]), dat["rolldev"], rtol=0, atol=1e-08)

dat["pitch"] = dat["pitch"].astype(np.float32)
dat["rolldev"] = dat["rolldev"].astype(np.float32)
dat["pitch", "rolldev"].write("pitch_roll.fits.gz", overwrite=True)
