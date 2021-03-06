#include <vu/ColorHelpers.h>

#include <map>

namespace vu {

std::vector<Vec3<unsigned char> > GetColorPalette(const int K) {

    using Color = Vec3<unsigned char>;

    static std::map<int, std::vector<Color> > prebakedPalettes;

    if (prebakedPalettes.empty()) {
        prebakedPalettes[3]  = { {141,211,199}, {255,255,179}, {190,186,218} };
        prebakedPalettes[4]  = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114} };
        prebakedPalettes[5]  = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211} };
        prebakedPalettes[6]  = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211}, {253,180,98} };
        prebakedPalettes[7]  = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211}, {253,180,98}, {179,222,105} };
        prebakedPalettes[8]  = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211}, {253,180,98}, {179,222,105}, {252,205,229} };
        prebakedPalettes[9]  = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211}, {253,180,98}, {179,222,105}, {252,205,229}, {217,217,21} };
        prebakedPalettes[10] = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211}, {253,180,98}, {179,222,105}, {252,205,229}, {217,217,21}, {188,128,189} };
        prebakedPalettes[11] = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211}, {253,180,98}, {179,222,105}, {252,205,229}, {217,217,21}, {188,128,189}, {204,235,197} };
        prebakedPalettes[12] = { {141,211,199}, {255,255,179}, {190,186,218}, {251,128,114}, {128,177,211}, {253,180,98}, {179,222,105}, {252,205,229}, {217,217,21}, {188,128,189}, {204,235,197}, {255,237,111} };
    }

    const auto it = prebakedPalettes.find(K);

    if (it == prebakedPalettes.cend()) {

        std::vector<Color> palette(K);
        for (int k = 0; k < K; ++k) {
            palette[k] = { static_cast<unsigned char>(128 + rand() % 128),
                           static_cast<unsigned char>(128 + rand() % 128),
                           static_cast<unsigned char>(128 + rand() % 128) };
        }

        return palette;

    } else {

        return it->second;

    }

}

} // namespace vu