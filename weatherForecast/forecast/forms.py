from django import forms

class WeatherForm(forms.Form):
    """
    Form for weather search functionality
    """
    city = forms.CharField(
        max_length=100, 
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'geo-input',
            'placeholder': 'Enter city name',
            'id': 'cityInput',
            'autocomplete': 'off'
        })
    ) 