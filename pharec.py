import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
    1- preprocess image
    2- use model to predict
"""

class Pharec:
    def __init__(self, model_path, image_size):
        self.class_names = [
            'absa',
            'adidas',
            'adobe',
            'airbnb',
            'alibaba',
            'aliexpress',
            'allegro',
            'amazon',
            'ameli_fr',
            'american_express',
            'anadolubank',
            'aol',
            'apple',
            'arnet_tech',
            'aruba',
            'att',
            'azul',
            'bahia',
            'banco_de_occidente',
            'banco_inter',
            'bankia',
            'barclaycard',
            'barclays',
            'bbt',
            'bcp',
            'bestchange',
            'blizzard',
            'bmo',
            'bnp_paribas',
            'bnz',
            'boa',
            'bradesco',
            'bt',
            'caixa_bank',
            'canada',
            'capital_one',
            'capitec',
            'cathay_bank',
            'cetelem',
            'chase',
            'cibc',
            'cloudconvert',
            'cloudns',
            'cogeco',
            'commonwealth_bank',
            'cox',
            'crate_and_barrel',
            'cryptobridge',
            'daum',
            'db',
            'dhl',
            'dkb',
            'docmagic',
            'dropbox',
            'ebay',
            'eharmony',
            'erste',
            'etisalat',
            'etrade',
            'facebook',
            'fibank',
            'file_transfer',
            'fnac',
            'fsnb',
            'godaddy',
            'google',
            'google_drive',
            'gov_uk',
            'grupo_bancolombia',
            'hfe',
            'hsbc',
            'htb',
            'icloud',
            'ics',
            'ieee',
            'impots_gov',
            'infinisource',
            'instagram',
            'irs',
            'itau',
            'itunes',
            'knab',
            'la_banque_postale',
            'la_poste',
            'latam',
            'lbb',
            'lcl',
            'linkedin',
            'lloyds_bank',
            'made_in_china',
            'mbank',
            'mdpd',
            'mew',
            'microsoft',
            'momentum_office_design',
            'ms_bing',
            'ms_office',
            'ms_onedrive',
            'ms_outlook',
            'ms_skype',
            'mweb',
            'my_cloud',
            'nab',
            'natwest',
            'navy_federal',
            'nedbank',
            'netflix',
            'netsons',
            'nordea',
            'ocn',
            'one_and_one',
            'orange',
            'orange_rockland',
            'otrs',
            'ourtime',
            'paschoalotto',
            'paypal',
            'postbank',
            'qnb',
            'rbc',
            'runescape',
            'sharp',
            'shoptet',
            'sicil_shop',
            'smartsheet',
            'smiles',
            'snapchat',
            'sparkasse',
            'standard_bank',
            'steam',
            'strato',
            'stripe',
            'summit_bank',
            'sunrise',
            'suntrust',
            'swisscom',
            'taxact',
            'tech_target',
            'telecom',
            'test_rite',
            'timeweb',
            'tradekey',
            'twins_bnk',
            'twitter',
            'typeform',
            'usaa',
            'walmart',
            'wells_fargo',
            'whatsapp',
            'wp60',
            'xtrix_tv',
            'yahoo',
            'youtube',
            'ziggo',
            'zoominfo'
        ];

        self.model_path = model_path
        self.image_size = image_size
        self.image_width = self.image_size[0]
        self.image_height = self.image_size[1]

        self.model = keras.models.load_model(model_path)

    def predict_domain(self, image):
        self.probits = tf.nn.softmax(
            self.model.predict(
                tf.keras.layers.Rescaling(
                    1./255,
                    input_shape=(self.image_width, self.image_height, 3)
                )(image)
            )
        ).numpy().squeeze(axis=0)
        self.pred_class = np.argmax(self.probits)
        return self.get_predicted_domain()

    def get_predicted_domain(self):
        self.pred_conf = np.around(self.probits[self.pred_class] * 100, 2)
        return self.class_names[self.pred_class], self.pred_conf

    def load_image(self, image_path): 
        return tf.expand_dims(
            keras.utils.img_to_array(
                keras.utils.load_img(
                        image_path,
                        target_size=self.image_size
                    )
            ),
            0
        )

