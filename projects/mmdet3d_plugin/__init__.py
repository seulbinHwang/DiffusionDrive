from .datasets import *
from .models import *
from .apis import *
from .core.evaluation import *
"""
from .datasets import * 등의 구문을 사용해서 
    하위 디렉토리(datasets, models, apis, core/evaluation)에 정의된 
    모든 공개(public) 심볼(함수, 클래스, 변수 등)을 현재 패키지의 네임스페이스에 포함시킴
즉, 외부에서 import projects.mmdet3d_plugin 또는 from projects.mmdet3d_plugin import SomeClass와 같이 사용할 때, 
    하위 모듈에서 __all__로 지정된 내용이나 공개된 항목들을 바로 접근할 수 있도록 만들어줍니다.
주의사항:
*를 사용한 임포트는 
    해당 모듈의 __all__에 정의된 항목만 가져오게 되는 경우도 있고, 
    __all__이 없으면 해당 모듈에서 언더스코어(_)로 시작하지 않는 모든 항목을 가져옵니다.
"""
