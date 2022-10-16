class TestLoadingLFFiles:
    def test_loading_existing_file_as_str_succeeds(self):
        pass

    def test_loading_existing_file_as_file_succeeds(self):
        pass

    def test_loading_not_existing_file_as_str_raises_file_not_found_error(self):
        pass

    def test_loading_not_existing_file_as_file_raises_file_not_found_error(self):
        pass

    # parametrize: field_type in [re,im,amplitude,intensity,phase]
    def test_loading_single_field_type_succeeds(self):
        pass

    # parametrize: field_types in [(re,im), (amplitude,phase), (intensity,phase)]
    def test_loading_supplemental_field_types_succeeds(self):
        pass

    # parametrize: field_types in [(re,amplitude), (re,phase), (re,intensity)...]
    def test_loading_not_supplemental_field_types_raises_value_error(self):
        pass

    # parametrize: field_types in [(re,im,phase)...]
    def test_loading_more_than_two_field_types_raises_value_error(self):
        pass


class TestLoadingPNGFiles:
    def test_loading_existing_file_as_str_succeeds(self):
        pass

    def test_loading_existing_file_as_file_succeeds(self):
        pass

    def test_loading_not_existing_file_as_str_raises_file_not_found_error(self):
        pass

    def test_loading_not_existing_file_as_file_raises_file_not_found_error(self):
        pass

    # parametrize: field_type in [re,im,amplitude,intensity,phase]
    def test_loading_single_field_type_succeeds(self):
        pass

    # parametrize: field_types in [(re,im), (amplitude,phase), (intensity,phase)]
    def test_loading_supplemental_field_types_succeeds(self):
        pass

    # parametrize: field_types in [(re,amplitude), (re,phase), (re,intensity)...]
    def test_loading_not_supplemental_field_types_raises_value_error(self):
        pass

    # parametrize: field_types in [(re,im,phase)...]
    def test_loading_more_than_two_field_types_raises_value_error(self):
        pass


class TestLoadingBMPFiles:
    def test_loading_existing_file_as_str_succeeds(self):
        pass

    def test_loading_existing_file_as_file_succeeds(self):
        pass

    def test_loading_not_existing_file_as_str_raises_file_not_found_error(self):
        pass

    def test_loading_not_existing_file_as_file_raises_file_not_found_error(self):
        pass

    # parametrize: field_type in [re,im,amplitude,intensity,phase]
    def test_loading_single_field_type_succeeds(self):
        pass

    # parametrize: field_types in [(re,im), (amplitude,phase), (intensity,phase)]
    def test_loading_supplemental_field_types_succeeds(self):
        pass

    # parametrize: field_types in [(re,amplitude), (re,phase), (re,intensity)...]
    def test_loading_not_supplemental_field_types_raises_value_error(self):
        pass

    # parametrize: field_types in [(re,im,phase)...]
    def test_loading_more_than_two_field_types_raises_value_error(self):
        pass
