# ruff: noqa: RUF001, RUF002
import re
from typing import ClassVar


class SpokenTextConverter:
    """
    A utility class for converting text containing numbers, dates, times, and currency
    into their spoken-word equivalents. Can you inagine how many edge cases you have to cover?

    This class provides methods to normalize and convert various text elements, such as:
    - Numbers (e.g., "3.14" → "three point one four")
    - Dates (e.g., "1/1/2024" → "one/one/twenty twenty-four")
    - Times (e.g., "3:00pm" → "three o'clock")
    - Currency (e.g., "$50.00" → "fifty dollars")
    - Percentages (e.g., "50%" → "fifty percent")
    - Titles and abbreviations (e.g., "Mr." → "Mister")
    - Years (e.g., "1999" → "nineteen ninety-nine")
    - Large numbers (e.g., "1000000" → "one million")
    - Decimals (e.g., "0.5" → "zero point five")
    - Mixed text (e.g., "The meeting is at 3:00pm on 1/1/2024.")
    - And more...


    Example usage:
        >>> converter = SpokenTextConverter()
        >>> result = converter.convert_to_spoken_text("The meeting is at 3:00pm on 1/1/2024.")
        >>> print(result)
        The meeting is at three o'clock on one/one/twenty twenty-four.
    """

    # Transliteration map for English to Russian (Cyrillic)
    TRANSLITERATION_MAP: ClassVar[dict[str, str]] = {
        'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
        'g': 'г', 'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л',
        'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'кв', 'r': 'р',
        's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
        'y': 'й', 'z': 'з',
        'A': 'А', 'B': 'Б', 'C': 'К', 'D': 'Д', 'E': 'Е', 'F': 'Ф',
        'G': 'Г', 'H': 'Х', 'I': 'И', 'J': 'Дж', 'K': 'К', 'L': 'Л',
        'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'Q': 'Кв', 'R': 'Р',
        'S': 'С', 'T': 'Т', 'U': 'У', 'V': 'В', 'W': 'В', 'X': 'Кс',
        'Y': 'Й', 'Z': 'З'
    }

    CONTRACTIONS: ClassVar[dict[str, str]] = {
        "I'm": "I am",
        "I'll": "I will",
        "I've": "I have",
        "I'd": "I would",
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'ll": " will",
        "'re": " are",
        "'ve": " have",
        "'m": " am",
        "'d": " would",
        "ain't": "is not",
    }

    def __init__(self) -> None:
        # Initialize any necessary state or configurations here, maybe for other languages?

        # Precompile quick check pattern
        # Note: Only check for mathematical operators that aren't commonly used in regular text
        """
        Initialize the SpokenTextConverter with regex patterns for identifying convertible text content.

        This method sets up a compiled regular expression pattern to quickly identify text
        that may require conversion, such as numbers, currency symbols, mathematical operators,
        common abbreviations, and ellipses.

        The pattern checks for:
        - Digits
        - Currency symbols ($ and £)
        - Specific mathematical operators (multiplication, division, exponentiation, roots)
        - Common title abbreviations
        - Ellipses (three or more dots, including spaced versions)

        The regex is compiled with verbose mode (re.VERBOSE) to allow more readable pattern construction.
        """
        self.convertible_pattern = re.compile(
            r"""(?x)
            \d                        # Any digit
            |\$|£                     # Currency symbols
            |[×÷^√∛]                 # Unambiguous mathematical operators (removed hyphen)
            |\b(?:Dr|Mr|Mrs|Ms)\.    # Common abbreviations
            |\.{3,}|\. \. \.         # Triple dots (including spaced version)
            """
        )

        # TODO: Add compiled regex patterns for other conversions

    def _number_to_words(self, num: float | str) -> str:
        """
        Convert a number into its spoken-word equivalent.

        Handles integers, floating-point numbers, and numeric strings, including:
        - Negative numbers
        - Large numbers (e.g., millions, billions)
        - Decimal numbers
        - Zero and whole numbers

        Parameters:
            num (float | str): The number to convert. Can be an integer, float, or numeric string.

        Returns:
            str: The spoken-word representation of the number.

        Raises:
            ValueError: If the input cannot be converted to a valid number.

        Examples:
            >>> converter._number_to_words(42)
            'forty-two'
            >>> converter._number_to_words(-17)
            'negative seventeen'
            >>> converter._number_to_words(1234567)
            'one million two hundred thirty-four thousand five hundred sixty-seven'
            >>> converter._number_to_words(3.14)
            'three point one four'
        """
        try:
            if isinstance(num, str):
                # Check if it's actually an integer in string form
                if "." not in num or num.endswith(".0"):
                    num = int(float(num))
                else:
                    num = float(num)

            # Special handling for integers
            if isinstance(num, int) or (isinstance(num, float) and num.is_integer()):
                num = int(num)  # Convert to int if it's a whole number

            if num == 0:
                return "zero"

            ones = [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]
            tens = [
                "",
                "",
                "twenty",
                "thirty",
                "forty",
                "fifty",
                "sixty",
                "seventy",
                "eighty",
                "ninety",
            ]
            scales = ["", "thousand", "million", "billion"]

            def process_chunk(n: int, scale: int) -> str:
                """
                Convert a chunk of a number into its spoken word representation.

                This method handles converting a three-digit number chunk into words, including handling
                hundreds, tens, and ones places. It supports numbers from 0 to 999 and can append
                scale words (thousand, million, etc.) when appropriate.

                Parameters:
                    n (int): The number chunk to convert (0-999)
                    scale (int): The scale index representing the magnitude (0 for ones, 1 for thousands,
                    2 for millions, etc.)

                Returns:
                    str: The spoken word representation of the number chunk, including optional scale word

                Example:
                    process_chunk(123, 1) returns "one hundred twenty-three thousand"
                    process_chunk(45, 0) returns "forty-five"
                """
                if n == 0:
                    return ""

                hundreds = n // 100
                remainder = n % 100

                words = []

                if hundreds > 0:
                    words.append(f"{ones[hundreds]} hundred")

                if remainder > 0:
                    if remainder < 20:
                        words.append(ones[remainder])
                    else:
                        tens_digit = remainder // 10
                        ones_digit = remainder % 10
                        if ones_digit == 0:
                            words.append(tens[tens_digit])
                        else:
                            words.append(f"{tens[tens_digit]}-{ones[ones_digit]}")

                if scale > 0 and len(words) > 0:
                    words.append(scales[scale])

                return " ".join(words)

            # Handle negative numbers
            if num < 0:
                return "negative " + self._number_to_words(abs(num))

            # Handle whole numbers differently from decimals
            if isinstance(num, int):
                if num == 0:
                    return "zero"

                intermediate_result: list[str] = []
                scale = 0

                while num > 0:
                    chunk = num % 1000
                    if chunk != 0:
                        chunk_words = process_chunk(chunk, scale)
                        intermediate_result.insert(0, chunk_words)
                    num //= 1000
                    scale += 1

                return " ".join(filter(None, intermediate_result))
            else:
                # Handle decimal numbers
                str_num = f"{num:.10f}".rstrip("0")  # Handle floating point precision
                if "." in str_num:
                    int_part, dec_part = str_num.split(".")
                else:
                    int_part, dec_part = str_num, ""

                int_num = int(int_part)

                # Convert integer part
                if int_num == 0:
                    result = "zero"
                else:
                    intermediate_result = []
                    scale = 0
                    while int_num > 0:
                        chunk = int_num % 1000
                        if chunk != 0:
                            chunk_words = process_chunk(chunk, scale)
                            intermediate_result.insert(0, chunk_words)
                        int_num //= 1000
                        scale += 1
                    result = " ".join(filter(None, intermediate_result))

                # Add decimal part if it exists
                if dec_part:
                    result = result + " point " + " ".join(ones[int(digit)] for digit in dec_part)
                return result
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid number format: {num}") from e

    def _split_num(self, num: re.Match) -> str:
        """
        Convert numbers, times, and years into their spoken-word equivalents.

        This method handles complex conversions for:
        - Time formats (12-hour and 24-hour)
            - With or without AM/PM
            - Handles "o'clock" for zero minutes
            - Converts minutes less than 10 with "oh"
        - Year formats
            - Single years (e.g., 1999)
            - Decades (e.g., 1950s)
            - Special handling for 2000 and 2000s
            - Supports plural forms for decades

        Parameters:
            num (re.Match): A regex match object containing a time, year, or number string.

        Returns:
            str: The spoken-word equivalent of the input time, year, or number.

        Raises:
            ValueError: If the input cannot be parsed as a valid time or number.
        """
        try:
            match_str = num.group()
            if ":" in match_str:
                # Split out any AM/PM first
                time_str = match_str.lower()
                am_pm = ""
                if "am" in time_str:
                    am_pm = " a m"
                    time_str = time_str.replace("am", "").strip()
                elif "pm" in time_str:
                    am_pm = " p m"
                    time_str = time_str.replace("pm", "").strip()

                try:
                    h, m = [int(n) for n in time_str.split(":")]
                    if not (0 <= h <= 23 and 0 <= m <= 59):
                        return match_str

                    # Handle minutes based on whether we have AM/PM
                    if m == 0:
                        if am_pm:  # If we have AM/PM, just use the hour
                            time = f"{self._number_to_words(h)}"
                        else:  # No AM/PM, use o'clock
                            time = f"{self._number_to_words(h)} o'clock"
                    elif m < 10:
                        time = f"{self._number_to_words(h)} oh {self._number_to_words(m)}"
                    else:
                        time = f"{self._number_to_words(h)} {self._number_to_words(m)}"

                    return f"{time}{am_pm}"

                except ValueError:
                    return match_str

            # Year handling
            try:
                number = int(match_str.rstrip("s"))  # Remove 's' if present
                if len(match_str) == 4 or (len(match_str) == 5 and match_str.endswith("s")):
                    left, right = divmod(number, 100)
                    s = "s" if match_str.endswith("s") else ""

                    # Special case for 2000 and 2000s
                    if number == 2000:
                        if s:
                            return "twenty hundreds"
                        else:
                            return "two thousand"
                    elif right == 0:
                        return f"{self._number_to_words(left)} hundred{s}"
                    elif right < 10:
                        return f"{self._number_to_words(left)} oh {self._number_to_words(right)}{s}"
                    else:
                        # Handle plural for decades (e.g., 1950s → "nineteen fifties")
                        if s and right >= 10:
                            decade_word = self._number_to_words(right).replace(" ", "-")
                            if decade_word.endswith("y"):
                                decade_word = decade_word[:-1] + "ies"
                            else:
                                decade_word += "s"
                            return f"{self._number_to_words(left)} {decade_word}"
                        return f"{self._number_to_words(left)} {self._number_to_words(right)}{s}"

                return self._number_to_words(number)
            except ValueError:
                return match_str
        except Exception:
            return num.group()

    def _flip_money(self, m: re.Match[str]) -> str:
        """
        Convert currency expressions into their spoken-word equivalents.

        Handles currency conversions for dollars and pounds, including whole numbers and decimal
        amounts. Supports singular and plural forms, and manages edge cases like zero cents/pence.

        Parameters:
            m (re.Match[str]): A regex match object containing a currency expression (e.g., "$50.00")

        Returns:
            str: The spoken-word representation of the currency amount

        Raises:
            ValueError: If the currency format is invalid or cannot be parsed

        Examples:
            "$5.00" → "five dollars"
            "$1.50" → "one dollar and fifty cents"
            "£10.00" → "ten pounds"
            "£1.01" → "one pound and one penny"
        """
        try:
            m = m.group()
            if not m or len(m) < 2:
                raise ValueError("Invalid currency format")

            bill = "dollar" if m[0] == "$" else "pound"
            amount_str = m[1:]

            if amount_str.isalpha():
                return f"{self._number_to_words(int(amount_str))} {bill}s"
            elif "." not in amount_str:
                amount = int(amount_str)
                s = "" if amount == 1 else "s"
                return f"{self._number_to_words(amount)} {bill}{s}"

            try:
                b, c = amount_str.split(".")
                if not b:  # Handle case like "$.50"
                    b = "0"
                s = "" if b == "1" else "s"
                c = int(c.ljust(2, "0"))

                # Don't add cents/pence if it's zero
                if c == 0:
                    return f"{self._number_to_words(int(b))} {bill}{s}"

                coins = f"cent{'' if c == 1 else 's'}" if m[0] == "$" else ("penny" if c == 1 else "pence")
                return f"{self._number_to_words(int(b))} {bill}{s} and {self._number_to_words(c)} {coins}"
            except ValueError as e:
                raise ValueError(f"Invalid currency format: {m}") from e
        except Exception:
            return m  # Return original text if conversion fails

    def _point_num(self, num: re.Match[str]) -> str:
        """
        Convert a decimal number to its spoken-word representation.

        Parameters:
            num (re.Match[str]): A regex match object containing a decimal number.

        Returns:
            str: The spoken-word equivalent of the decimal number.

        Converts the matched decimal number to a float and uses the _number_to_words method
        to generate its spoken representation.
        """
        return self._number_to_words(float(num.group()))

    def _convert_percentages(self, text: str) -> str:
        """
        Convert percentage expressions in the text to their spoken-word equivalents.

        This method uses regular expressions to identify percentage values and converts them to their spoken
        form. It handles both whole numbers and decimal percentages, converting the numeric value to words
        followed by the word "percent".

        Parameters:
            text (str): The input text containing percentage expressions (e.g., "50%").

        Returns:
            str: The input text with percentages converted to spoken words.

        Examples:
            >>> converter = SpokenTextConverter()
            >>> converter._convert_percentages("The stock rose 25%")
            'The stock rose twenty-five percent'
            >>> converter._convert_percentages("Accuracy is 99.5%")
            'Accuracy is ninety-nine point five percent'
        """

        def replace_match(match: re.Match) -> str:
            """
            Convert a regex match of a percentage to its spoken word representation.

            Parameters:
                match (re.Match): A regex match object containing a percentage value.

            Returns:
                str: The spoken word representation of the percentage, including the word "percent".

            Raises:
                ValueError: If the matched number cannot be converted to a numeric type.
            """
            number = match.group(1)
            # Handle whole numbers without decimal point
            if "." not in number:
                return f"{self._number_to_words(int(number))} percent"
            return f"{self._number_to_words(float(number))} percent"

        return re.sub(r"(\d+\.?\d*)%", replace_match, text)

    def _contains_convertible_content(self, text: str) -> bool:
        """
        Fast check if text contains any content that needs conversion.
        Only looks for unambiguous indicators of convertible content.
        """
        return bool(self.convertible_pattern.search(text))

    def _is_russian_text(self, text: str) -> bool:
        """
        Detect if the text is predominantly Russian (Cyrillic).

        Returns True if more than 50% of alphabetic characters are Cyrillic.
        """
        cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin_count = sum(1 for c in text if c.isalpha() and c.isascii())

        total = cyrillic_count + latin_count
        if total == 0:
            return False

        return cyrillic_count / total > 0.5

    def _transliterate_to_cyrillic(self, text: str) -> str:
        """
        Transliterate English words to Cyrillic for Russian TTS.

        This allows Russian TTS to pronounce English words phonetically.
        Example: "mistook" → "мистук", "comedy" → "комедй"
        """
        # Only transliterate Latin words, keep everything else
        def transliterate_word(match: re.Match) -> str:
            word = match.group(0)
            # Check if word is all Latin letters
            if all(c.isalpha() and c.isascii() or c in "'-" for c in word):
                # Transliterate letter by letter
                result = []
                for char in word:
                    if char in self.TRANSLITERATION_MAP:
                        result.append(self.TRANSLITERATION_MAP[char])
                    else:
                        result.append(char)  # Keep punctuation as is
                return ''.join(result)
            return word

        # Match words (sequences of letters, hyphens, and apostrophes)
        return re.sub(r"[A-Za-z][-'A-Za-z]*", transliterate_word, text)

    def _number_to_russian_words(self, num: float | str) -> str:
        """
        Convert a number to Russian words.

        Examples:
            42 → "сорок два"
            3.14 → "три точка один четыре"
            -17 → "минус семнадцать"
        """
        try:
            if isinstance(num, str):
                if "." not in num or num.endswith(".0"):
                    num = int(float(num))
                else:
                    num = float(num)

            if isinstance(num, int) or (isinstance(num, float) and num.is_integer()):
                num = int(num)

            if num == 0:
                return "ноль"

            ones = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
            teens = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
                     "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
            tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
                    "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
            hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
                        "шестьсот", "семьсот", "восемьсот", "девятьсот"]

            # Handle negative
            if num < 0:
                return "минус " + self._number_to_russian_words(abs(num))

            # Handle decimal
            if isinstance(num, float):
                str_num = f"{num:.10f}".rstrip("0")
                if "." in str_num:
                    int_part, dec_part = str_num.split(".")
                    result = self._number_to_russian_words(int(int_part))
                    result += " точка " + " ".join(ones[int(d)] if int(d) < len(ones) else str(d) for d in dec_part)
                    return result

            # Handle integers up to 999
            if num < 10:
                return ones[num]
            elif num < 20:
                return teens[num - 10]
            elif num < 100:
                tens_digit = num // 10
                ones_digit = num % 10
                return (tens[tens_digit] + (" " + ones[ones_digit] if ones_digit else "")).strip()
            elif num < 1000:
                hundreds_digit = num // 100
                remainder = num % 100
                result = hundreds[hundreds_digit]
                if remainder:
                    result += " " + self._number_to_russian_words(remainder)
                return result.strip()
            else:
                # For larger numbers, fall back to digit-by-digit
                return " ".join(ones[int(d)] if int(d) < len(ones) else str(d) for d in str(num))

        except (ValueError, TypeError):
            return str(num)

    def _convert_mathematical_notation(self, text: str) -> str:
        """
        Convert mathematical notation to spoken form.

        Converts various mathematical symbols and notations into their spoken word equivalents.
        Handles exponents, roots, arithmetic operations, fractions, and comparison symbols.

        Parameters:
            text (str): Text containing mathematical notation to be converted

        Returns:
            str: Text with mathematical notation transformed into spoken language

        Handles conversions such as:
            - Exponents: "8^2" → "eight to the power of two"
            - Square roots: "√9" → "square root of nine"
            - Cube roots: "∛8" → "cube root of eight"
            - Basic operations: "5 + 3" → "five plus three"
            - Fractions: "1/2" → "one over two"
            - Equals signs: "=" → "equals"
            - Multiplication: "×" → "times"
            - Division: "÷" → "divided by"

        Notes:
            - Preserves date-like fractions (e.g., 1/1/2024)
            - Handles both numeric and letter variable exponents
            - Cleans up extra whitespace after conversion
        """

        # Fast fail if no convertible content
        if not self._contains_convertible_content(text):
            return text

        # Helper function to convert numbers in matched patterns
        def convert_numbers_in_match(match: re.Match, pattern: str) -> str:
            """
            Convert numeric parts of a regex match to their spoken word representation.

            Parameters:
                match (re.Match): A regex match object containing numeric groups
                pattern (str): A formatting pattern to reconstruct the matched text with converted numbers

            Returns:
                str: A string with numeric groups replaced by their spoken word equivalents

            Notes:
                - Iterates through match groups and converts digit-only groups to words
                - Uses the instance method `_number_to_words` for number conversion
                - Preserves non-numeric groups in their original form
            """
            parts = list(match.groups())
            for i, part in enumerate(parts):
                if part and part.isdigit():
                    parts[i] = self._number_to_words(int(part))
            return pattern.format(*parts)

        # Convert basic arithmetic symbols first
        text = text.replace(" = ", " equals ")
        text = text.replace("=", " equals ")
        text = text.replace(" + ", " plus ")
        text = text.replace("+", " plus ")
        text = text.replace(" - ", " minus ")
        text = text.replace(" × ", " times ")
        text = text.replace("×", " times ")
        text = text.replace(" ÷ ", " divided by ")
        text = text.replace("÷", " divided by ")

        # Convert exponents (e.g., 8^2, x^2, etc.)
        text = re.sub(
            r"(\d+)\^(\d+)",
            lambda m: convert_numbers_in_match(m, "{0} to the power of {1}"),
            text,
        )

        # Convert letter variables with exponents (e.g., x^2)
        text = re.sub(
            r"([a-zA-Z])\^(\d+)",
            lambda m: f"{m.group(1)} to the power of {self._number_to_words(int(m.group(2)))}",
            text,
        )

        # Convert square roots (√)
        text = re.sub(
            r"√(\d+)",
            lambda m: f"square root of {self._number_to_words(int(m.group(1)))}",
            text,
        )

        # Convert cube roots (∛)
        text = re.sub(
            r"∛(\d+)",
            lambda m: f"cube root of {self._number_to_words(int(m.group(1)))}",
            text,
        )

        # Convert mathematical fractions (only if not part of a date)
        def convert_fraction(match: re.Match) -> str:
            # Skip if it looks like a date (e.g., 1/1/2024)
            """
            Convert a fraction match to its spoken word representation.

            This method handles fraction conversion, avoiding date-like patterns and converting
            the numerator and denominator to their word equivalents.

            Parameters:
                match (re.Match): A regex match object representing a fraction.

            Returns:
                str: A spoken word representation of the fraction, in the format "numerator over denominator".

            Example:
                "3/4" becomes "three over four"
                "1/2" becomes "one over two"

            Notes:
                - Skips conversion for patterns that look like dates (e.g., 1/1/2024)
                - Uses _number_to_words method to convert numeric parts to words
            """
            if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", match.group(0)):
                return match.group(0)
            num = self._number_to_words(int(match.group(1)))
            den = self._number_to_words(int(match.group(2)))
            return f"{num} over {den}"

        text = re.sub(r"(\d+)/(\d+)(?!/)", convert_fraction, text)

        # Clean up any extra spaces that may have been introduced
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def text_to_spoken(self, text: str) -> str:
        """
        Convert a given text into its spoken-word equivalent.

        This method processes the input text through multiple stages of transformation, including:
        1. Expanding contractions
        2. Normalizing quotes and punctuation
        3. Converting titles and abbreviations
        4. Preparing number formatting
        5. Converting dates, mathematical notation, percentages, currency, times, years, and numbers

        The conversion handles various text elements like numbers, dates, times, currency, and percentages,
        transforming them into their spoken-word representations.

        For Russian text (detected by Cyrillic characters):
        - Transliterates English words to Cyrillic for proper TTS pronunciation
        - Converts numbers to Russian words

        Args:
            text (str): The input text to convert.

        Returns:
            str: The input text with numbers, dates, times, and currency converted to spoken words.

        Raises:
            ValueError: If the input text contains invalid or unsupported formats during conversion.

        Notes:
            - Preserves acronyms and special cases like "I"
            - Handles complex number formats including large numbers and decimals
            - Supports multiple currency symbols and percentage conversions
        """
        # Detect if text is Russian
        is_russian = self._is_russian_text(text)

        # If Russian, transliterate English words first
        if is_russian:
            text = self._transliterate_to_cyrillic(text)
            # Convert percentages to Russian
            text = re.sub(r"(\d+\.?\d*)%", lambda m: f"{self._number_to_russian_words(m.group(1))} процентов", text)
            # Convert standalone numbers to Russian
            text = re.sub(r"\b\d+\.?\d*\b", lambda m: self._number_to_russian_words(m.group()), text)
            return text.strip()

        # English processing (original logic)
        # 1. First expand contractions (this part works correctly)
        for contraction, expansion in sorted(self.CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(contraction, expansion)

        # remove leading and trailing whitespace and empty lines
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        # 2. Quote normalization
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        text = text.replace("(", "«").replace(")", "»")
        # Handle square brackets - replace with spaces for cleaner TTS output
        text = text.replace("[", " ").replace("]", " ")

        # 3. Punctuation normalization
        # a. Replace common punctuation marks
        for a, b in zip("、。！，：；？", ",.!,:;?", strict=False):
            text = text.replace(a, b + " ")

        # b. Remove ellipses
        text = re.sub(r"\.{3,}|\. \. \.", "", text)

        # 4. Whitespace normalization
        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)

        # 5. Convert titles and abbreviations
        text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
        text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
        text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
        text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
        text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)

        # Convert mixed case words to lowercase unless they're acronyms
        def process_word(match: re.Match) -> str:
            """
            Converts a matched word to its spoken form while preserving specific capitalization rules.

            This method handles word conversion with special considerations:
            - Acronyms (all uppercase words with length > 1) are split into individual letters
            - The word "I" is preserved in its uppercase form
            - Other words are converted to lowercase

            Parameters:
                match (re.Match): A regex match object containing the word to be processed

            Returns:
                str: The processed word according to the specified capitalization rules
            """
            word = match.group(0)
            # Keep uppercase if it's an acronym (all caps and length > 1)
            if word.isupper() and len(word) > 1:
                return " ".join(word)  # Split into individual letters
            # Special case: preserve "I" as uppercase
            if word == "I":
                return word
            return word.lower()

        text = re.sub(r"\b[A-Za-z]+\b", process_word, text)

        # 6. Number formatting preparation
        # Remove commas in numbers but preserve them for later conversion
        def preserve_large_numbers(match: re.Match) -> str:
            """
            Convert a matched large number (with commas) to its spoken word representation.

            Parameters:
                match (re.Match): A regex match object containing a large number with comma separators.

            Returns:
                str: The spoken word representation of the number.

            Notes:
                - Removes commas from the matched number before conversion
                - Uses the class's _number_to_words method to convert the number
                - Handles large numbers by converting them to integers first
            """
            num = int(match.group().replace(",", ""))
            return self._number_to_words(num)

        text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", preserve_large_numbers, text)
        text = re.sub(r"(?<=\d),(?=\d)", "", text)

        # 7. Remove AM/PM but preserve the time part
        # text = re.sub(r"(\d+:\d+)\s*(?:am|pm)\b", r"\1", text, flags=re.IGNORECASE)

        # 8. Date conversion (before other number conversions)
        def convert_date(match: re.Match) -> str:
            """
            Convert a date match object into its spoken-word representation.

            This method handles date formatting by converting numeric date components
            (month, day, year) into their spoken-word equivalents. It supports two primary
            formats:
            - Standard date format with a 4-digit year (MM/DD/YYYY)
            - Shorter date formats with 2-digit components

            Parameters:
                match (re.Match): A regex match object containing a date string

            Returns:
                str: A spoken-word representation of the date, with numeric components
                     converted to words

            Examples:
                - "12/25/2000" → "twelve/twenty-five/two thousand"
                - "1/1/23" → "one/one/twenty-three"
            """
            parts = match.group().split("/")
            if len(parts) == 3 and len(parts[2]) == 4:
                # Convert the year part separately
                year = int(parts[2])
                if year == 2000:
                    year_text = "two thousand"
                else:
                    left, right = divmod(year, 100)
                    if right == 0:
                        year_text = f"{self._number_to_words(left)} hundred"
                    else:
                        year_text = f"{self._number_to_words(left)} {self._number_to_words(right)}"
                return f"{self._number_to_words(int(parts[0]))}/{self._number_to_words(int(parts[1]))}/{year_text}"
            return "/".join(self._number_to_words(int(part)) for part in parts)

        text = re.sub(r"\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b", convert_date, text)

        # 9. Convert mathematical notation (before other number conversions)
        text = self._convert_mathematical_notation(text)

        # 10. Number conversions in specific order:
        # a. Percentages first
        text = self._convert_percentages(text)

        # b. Currency
        text = re.sub(
            r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
            self._flip_money,
            text,
        )

        # c. Times
        text = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*(?:am|pm))?\b", self._split_num, text, flags=re.IGNORECASE)

        # d. Years - The key fix is to use lambda to return string directly
        text = re.sub(
            r"\b\d{4}s?\b",
            lambda m: self._split_num(m),
            text,
        )

        # e. Decimal numbers
        text = re.sub(r"\d*\.\d+", self._point_num, text)

        # f. Standalone integers (new addition)
        text = re.sub(r"\b\d+\b", lambda m: self._number_to_words(int(m.group())), text)

        # 10. Final formatting
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
        text = re.sub(r"(?<=\d)S", " S", text)
        text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
        text = re.sub(r"(?<=X')S\b", "s", text)
        text = re.sub(r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text)
        text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)

        # 11. Final cleanup
        # text = re.sub(r"\b(?:am|pm)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"  +", " ", text)  # Clean up any double spaces that may have been created

        return text.strip()
