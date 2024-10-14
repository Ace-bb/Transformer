from conf.settings import *
from utils.utils import subsequent_mask

def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    alt_draw = alt.Chart(LS_data).mark_rect().properties(height=250, width=250).encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        ).interactive()
    print(alt_draw)
    alt_draw.save("subsequent_mask.html")


example_mask()