### src/utils/conversation.py
from dataclasses import dataclass
from enum import auto, Enum
import json

from PIL.Image import Image
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:\n'


class Role(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()
    INTERPRETER = auto()
    OBSERVATION = auto()

    def __str__(self):
        match self:
            case Role.SYSTEM:
                return "<|system|>"
            case Role.USER:
                return "<|user|>"
            case Role.ASSISTANT | Role.TOOL | Role.INTERPRETER:
                return "<|assistant|>"
            case Role.OBSERVATION:
                return "<|observation|>"

    # Get the message block for the given role
    def get_message(self):
        # Compare by value here, because the enum object in the session state
        # is not the same as the enum cases here, due to streamlit's rerunning
        # behavior.
        match self.value:
            case Role.SYSTEM.value:
                return
            case Role.USER.value:
                return st.chat_message(name="user", avatar="user")
            case Role.ASSISTANT.value:
                return st.chat_message(name="assistant", avatar="assistant")
            case Role.TOOL.value:
                return st.chat_message(name="tool", avatar="assistant")
            case Role.INTERPRETER.value:
                return st.chat_message(name="interpreter", avatar="assistant")
            case Role.OBSERVATION.value:
                return st.chat_message(name="observation", avatar="user")
            case _:
                st.error(f'Unexpected role: {self}')


@dataclass
class Conversation:
    role: Role
    content: str
    tool: str | None = None
    image: Image | None = None

    def __str__(self) -> str:
        print(self.role, f"{self.content}", self.tool)
        match self.role:
            case Role.SYSTEM | Role.USER | Role.ASSISTANT | Role.OBSERVATION:
                return f'{self.role}\n{self.content}'
            case Role.TOOL:
                return f'{self.role}{self.tool}\n{self.content}'
            case Role.INTERPRETER:
                return f'{self.role}interpreter\n{self.content}'

    # Human readable format
    def get_text(self) -> str:
        # text = postprocess_text(self.content)
        text = self.content  # 只考虑文本情况
        match self.role.value:
            case Role.TOOL.value:
                text = f'Calling tool `{self.tool}`:\n\n{text}'
            case Role.INTERPRETER.value:
                text = f'{text}'
            case Role.OBSERVATION.value:
                text = f'Observation:\n```\n{text}\n```'
        return text

    # Display as a markdown block
    def show(self, placeholder: DeltaGenerator | None = None) -> str:
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()
        # if self.image:
        #     message.image(self.image)
        # else:
        #     text = self.get_text()
        #     message.markdown(text)
        if isinstance(self.content, (list, tuple)):
            for content in self.content:
                message.markdown(content)
        else:
            message.write(self.content)

    def to_dict(self):
        convers = []
        if isinstance(self.content, (list, tuple)):
            for c in self.content:
                convers.append({"role": f"{self.role}", "content": f"{c}"})
        else:
            convers.append({"role": f"{self.role}", "content": f"{self.content}"})
        return convers