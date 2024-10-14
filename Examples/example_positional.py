from conf.settings import *
from Embedding.PositionalEncoding import PositionalEncoding

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    draw_img = alt.Chart(data).mark_line().properties(width=800).encode(x="position", y="embedding", color="dimension:N").interactive()
    draw_img.save("positional_encoding.html")

example_positional()