import argparse
import dataclasses
from dataclasses import fields
from typing import Type, TypeVar, Any, Union, get_origin, get_args

T = TypeVar("T")

def build_parser_from_dataclass(dc: Type[T], description: str = "") -> argparse.ArgumentParser:
    """
    Dynamically constructs an ArgumentParser instance mapped directly to dataclass fields.
    Supports automatic type inference for primitives, explicit sequences, boolean flags, 
    and Optional types safely.
    """
    parser = argparse.ArgumentParser(
        description=description, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    for field in fields(dc):
        # Convert snake_case internal attributes to standard CLI kebab-case flags
        arg_name = f"--{field.name.replace('_', '-')}"
        
        # Enforce type-safe operational boundaries for missing values
        default_val = field.default
        if default_val is dataclasses.MISSING:
            if field.default_factory is not dataclasses.MISSING:
                default_val = field.default_factory()
            else:
                default_val = None

        field_type = field.type
        origin = get_origin(field_type)
        
        # Handle Optional or Union types (e.g., Optional[int], int | None) gracefully
        if origin is Union:
            type_args = get_args(field_type)
            # Filter out NoneType to find the actual underlying type
            actual_types = [t for t in type_args if t is not type(None)]
            if len(actual_types) == 1:
                field_type = actual_types[0]
                origin = get_origin(field_type)
        
        # Determine argument action based on inferred types
        if origin is list:
            inner_type = get_args(field_type)[0]
            parser.add_argument(arg_name, nargs="+", type=inner_type, default=default_val)
        elif field_type is bool:
            parser.add_argument(arg_name, default=default_val, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(arg_name, type=field_type, default=default_val)
            
    return parser

def parse_config(dc: Type[T], description: str = "") -> T:
    """
    Parses command-line utility flags and instantiates the target configuration dataclass instance.
    """
    parser = build_parser_from_dataclass(dc, description)
    args = parser.parse_args()
    return dc(**vars(args))