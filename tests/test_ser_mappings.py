"""各データセットの感情ラベルを共通クラスへ対応付ける契約を検証する。"""

import unittest

from ser_pipeline.contracts import (
    CLASS_TO_INDEX,
    FEATURE_LAYER,
    LABEL_ORDER,
    load_mapping_config,
    map_emotion,
    normalize_layer,
)


class SerMappingTest(unittest.TestCase):
    def test_label_order_and_all_configured_source_labels(self):
        self.assertEqual(LABEL_ORDER, ("anger", "happy", "sadness", "disgust"))
        self.assertEqual(CLASS_TO_INDEX, {label: index for index, label in enumerate(LABEL_ORDER)})
        config = load_mapping_config()
        expected = {
            "msp_podcast": ({"A", "H", "S", "D"}, {"C", "F", "N", "O", "U", "X"}),
            "hcudb1": (
                {"怒り", "狂喜・楽しい", "余裕・嬉しい", "憂鬱・悲しい", "嫌い"},
                {"驚き", "恐れ", "冷静", "軽蔑", "リラックス・気楽", "眠い・疲れた"},
            ),
            "iemocap": ({"ang", "hap", "exc", "sad", "dis"}, {"fea", "fru", "neu", "oth", "sur", "xxx"}),
        }
        for dataset, (included, excluded) in expected.items():
            self.assertEqual(set(config["datasets"][dataset]["mappings"]), included)
            self.assertEqual(set(config["datasets"][dataset]["excluded_labels"]), excluded)
            for label in included:
                decision = map_emotion(dataset, label)
                self.assertTrue(decision.included)
                self.assertEqual(decision.class_index, CLASS_TO_INDEX[decision.mapped_emotion])
            for label in excluded:
                decision = map_emotion(dataset, label)
                self.assertFalse(decision.included)
                self.assertIsNone(decision.class_index)
                self.assertEqual(decision.exclusion_reasons, ("label_not_in_primary_4",))

    def test_hcudb_disgust_mapping_is_the_only_approximation(self):
        self.assertTrue(map_emotion("hcudb1", "嫌い").approximate_mapping)
        for label in ("怒り", "狂喜・楽しい", "余裕・嬉しい", "憂鬱・悲しい"):
            self.assertFalse(map_emotion("hcudb1", label).approximate_mapping)

    def test_unknown_label_and_intermediate_layer_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            map_emotion("iemocap", "made-up")
        self.assertEqual(normalize_layer("final"), FEATURE_LAYER)
        for layer in (0, 11, "11", FEATURE_LAYER):
            with self.assertRaisesRegex(ValueError, "only 'final'"):
                normalize_layer(layer)


if __name__ == "__main__":
    unittest.main()
