from __future__ import annotations
import pyparsing as pyp
from . import type_parsring as tp

from ..objects.value import (
    NullValue,
    IntValue,
    LongValue,
    DoubleValue,
    BoolValue,
    CharValue,
    StringValue,
    ListValue,
    DictValue,
)

from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.objects.typed_value import get_typed_value, TypedValue

pyp_value = None

import re
def get_pyp_value():
    global pyp_value
    if pyp_value is not None:
        return pyp_value
    data_type = tp.get_pyp_type()
    typed_value = pyp.Forward()

    null_literal = pyp.Suppress("null")
    double_literal = pyp.Regex(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?")

    # Ensuring that at least one of the optional parts must be present
    double_literal.add_condition(
        lambda tokens: bool(re.search(r'\.\d+|[eE][+-]?\d+', tokens[0]))
    )
    # double_literal.add_condition(
    #     lambda s, loc, tokens: bool(re.search(r'\.\d+|[eE][+-]?\d+', tokens[0]))
    # )
    nan_literal = pyp.Literal("nan")
    inf_literal = pyp.Combine(pyp.Optional("-") + pyp.Literal("inf"))
    double_literal = nan_literal | inf_literal | double_literal
    integer_literal = pyp.Combine(pyp.Optional("-") + pyp.Word(pyp.nums)) + ~(
        pyp.FollowedBy("L") | pyp.FollowedBy(".") | pyp.FollowedBy("e") | pyp.FollowedBy("E")
    )
    long_literal = pyp.Combine(pyp.Optional("-") + pyp.Word(pyp.nums)) + pyp.Suppress(
        "L"
    )

    # double_literal = pyp.Combine(
    #     pyp.Optional("-") + pyp.Word(pyp.nums) + "." + pyp.Word(pyp.nums)
    # )


    # double_literal.add_parse_action(lambda t: print(t[0]))
    char_literal = pyp.QuotedString(
        quoteChar="'",
        escChar="\\",
        unquoteResults=False,
        multiline=True,
        convertWhitespaceEscapes=False,
    )
    bool_literal = pyp.Literal("true") | pyp.Literal("false")
    string_literal = pyp.QuotedString(
        quoteChar='"',
        escChar="\\",
        unquoteResults=False,
        multiline=True,
        convertWhitespaceEscapes=False,
    )

    null_literal.set_parse_action(lambda t: NullValue())
    integer_literal.set_parse_action(lambda t: IntValue(int(t[0])))
    long_literal.set_parse_action(lambda t: LongValue(int(t[0])))
    double_literal.set_parse_action(lambda t: DoubleValue(float(t[0])))
    bool_literal.set_parse_action(lambda t: BoolValue(bool(t[0] == "true")))
    char_literal.set_parse_action(lambda t: CharValue(t[0]))
    string_literal.set_parse_action(lambda t: StringValue(t[0]))

    list_literal = (
        pyp.Suppress("[")
        + pyp.Optional(pyp.delimitedList(typed_value))
        + pyp.Suppress("]")
    )
    kv_pair = typed_value + pyp.Suppress("=>") + typed_value
    dict_literal = (
        pyp.Suppress("{") + pyp.Optional(pyp.delimitedList(kv_pair)) + pyp.Suppress("}")
    )

    list_literal.set_parse_action(lambda t: ListValue(list(t)))
    dict_literal.set_parse_action(lambda t: DictValue(list(t)))

    value = (
        null_literal
        | double_literal
        | integer_literal
        | long_literal
        | bool_literal
        | char_literal
        | string_literal
        | list_literal
        | dict_literal
    )

    typed_value << (value + pyp.Suppress(":") + data_type | value)

    typed_value.set_parse_action(
        lambda t: get_typed_value(t[0], (t[1] if len(t) == 2 else None))
    )

    pyp_value = typed_value

    return typed_value

def get_pyp_value():
    global pyp_value
    if pyp_value is not None:
        return pyp_value
    data_type = tp.get_pyp_type()
    typed_value = pyp.Forward()

    null_literal = pyp.Suppress("null")
    double_literal = pyp.Regex(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?")

    # Ensuring that at least one of the optional parts must be present
    double_literal.add_condition(
        lambda tokens: bool(re.search(r'\.\d+|[eE][+-]?\d+', tokens[0]))
    )
    # double_literal.add_condition(
    #     lambda s, loc, tokens: bool(re.search(r'\.\d+|[eE][+-]?\d+', tokens[0]))
    # )
    nan_literal = pyp.Literal("nan")
    inf_literal = pyp.Combine(pyp.Optional("-") + pyp.Literal("inf"))
    double_literal = nan_literal | inf_literal | double_literal
    
    # Prioritize double parsing by using 'double_literal' first
    integer_literal = (
        pyp.Combine(pyp.Optional("-") + pyp.Word(pyp.nums))
        + ~(
            pyp.FollowedBy("L")
            | pyp.FollowedBy(".")
            | pyp.FollowedBy("e")
            | pyp.FollowedBy("E")
        )
    )
    long_literal = pyp.Combine(pyp.Optional("-") + pyp.Word(pyp.nums)) + pyp.Suppress(
        "L"
    )

    char_literal = pyp.QuotedString(
        quoteChar="'",
        escChar="\\",
        unquoteResults=False,
        multiline=True,
        convertWhitespaceEscapes=False,
    )
    bool_literal = pyp.Literal("true") | pyp.Literal("false")
    string_literal = pyp.QuotedString(
        quoteChar='"',
        escChar="\\",
        unquoteResults=False,
        multiline=True,
        convertWhitespaceEscapes=False,
    )

    # Set parse actions to map parsed tokens to custom objects
    null_literal.set_parse_action(lambda t: NullValue())
    double_literal.set_parse_action(lambda t: DoubleValue(float(t[0])))
    integer_literal.set_parse_action(lambda t: IntValue(int(t[0])))
    long_literal.set_parse_action(lambda t: LongValue(int(t[0])))
    bool_literal.set_parse_action(lambda t: BoolValue(t[0] == "true"))
    char_literal.set_parse_action(lambda t: CharValue(t[0]))
    string_literal.set_parse_action(lambda t: StringValue(t[0]))

    list_literal = (
        pyp.Suppress("[")
        + pyp.Optional(pyp.delimitedList(typed_value))
        + pyp.Suppress("]")
    )
    kv_pair = typed_value + pyp.Suppress("=>") + typed_value
    dict_literal = (
        pyp.Suppress("{") + pyp.Optional(pyp.delimitedList(kv_pair)) + pyp.Suppress("}")
    )

    list_literal.set_parse_action(lambda t: ListValue(list(t)))
    dict_literal.set_parse_action(lambda t: DictValue(list(t)))

    # Updated parsing priority: double_literal is checked before integer_literal
    value = (
        null_literal
        | double_literal
        | integer_literal
        | long_literal
        | bool_literal
        | char_literal
        | string_literal
        | list_literal
        | dict_literal
    )

    typed_value << (value + pyp.Suppress(":") + data_type | value)

    typed_value.set_parse_action(
        lambda t: get_typed_value(t[0], (t[1] if len(t) == 2 else None))
    )

    pyp_value = typed_value

    return typed_value

def value_parsing(s: str) -> TypedValue:
    data_value = get_pyp_value().parse_string(s, parse_all=True)
    return data_value[0]
