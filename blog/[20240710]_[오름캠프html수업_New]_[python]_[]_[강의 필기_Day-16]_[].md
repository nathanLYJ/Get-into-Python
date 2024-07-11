<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Document</title>
</head>

<style>
 table {
   border-collapse: collapse;
   width: 100%;
 }
 th, td {
   border: 1px solid #ddd;
   padding: 8px;
   text-align: left;
   white-space: nowrap;
 }
 th {
   background-color: #f2f2f2;
 }
  </style>
  <body>

  <table>
 <tr>
   <th>Block</th>
   <th>사용 상황</th>
   <th>용법</th>
   <th>Inline</th>
   <th>사용 상황</th>
   <th>용법</th>
   <th>Inline-block</th>
   <th>사용 상황</th>
   <th>용법</th>
 </tr>
 <tr>
   <td>&lt;div&gt;</td>
   <td>여러 요소를 그룹화하거나 레이아웃을 구성할 때</td>
   <td>&lt;div&gt;내용&lt;/div&gt;</td>
   <td>&lt;span&gt;</td>
   <td>텍스트의 일부를 묶어 스타일을 적용하거나 조작할 때</td>
   <td>&lt;span&gt;텍스트&lt;/span&gt;</td>
   <td>&lt;button&gt;</td>
   <td>사용자가 클릭할 수 있는 버튼을 만들 때</td>
   <td>&lt;button&gt;클릭&lt;/button&gt;</td>
 </tr>
 <tr>
   <td>&lt;p&gt;</td>
   <td>일반 텍스트 단락을 나타낼 때</td>
   <td>&lt;p&gt;문단 내용&lt;/p&gt;</td>
   <td>&lt;a&gt;</td>
   <td>다른 페이지나 같은 페이지의 특정 부분으로 연결할 때</td>
   <td>&lt;a href="url"&gt;링크&lt;/a&gt;</td>
   <td>&lt;input&gt;</td>
   <td>사용자로부터 다양한 유형의 입력을 받을 때</td>
   <td>&lt;input type="text"&gt;</td>
 </tr>
 <tr>
   <td>&lt;h1&gt;-&lt;h6&gt;</td>
   <td>페이지 구조를 나타내는 제목을 정의할 때</td>
   <td>&lt;h1&gt;제목&lt;/h1&gt;</td>
   <td>&lt;strong&gt;</td>
   <td>텍스트를 강조하여 중요성을 나타낼 때</td>
   <td>&lt;strong&gt;중요&lt;/strong&gt;</td>
   <td>&lt;select&gt;</td>
   <td>드롭다운 선택 메뉴를 만들 때</td>
   <td>&lt;select&gt;&lt;option&gt;옵션&lt;/option&gt;&lt;/select&gt;</td>
 </tr>
 <tr>
   <td>&lt;ul&gt;</td>
   <td>순서가 없는 목록을 만들 때</td>
   <td>&lt;ul&gt;&lt;li&gt;항목&lt;/li&gt;&lt;/ul&gt;</td>
   <td>&lt;em&gt;</td>
   <td>텍스트에 강조 의미를 부여할 때</td>
   <td>&lt;em&gt;강조&lt;/em&gt;</td>
   <td>&lt;textarea&gt;</td>
   <td>여러 줄의 텍스트 입력이 필요할 때</td>
   <td>&lt;textarea&gt;&lt;/textarea&gt;</td>
 </tr>
 <tr>
   <td>&lt;ol&gt;</td>
   <td>순서가 있는 목록을 만들 때</td>
   <td>&lt;ol&gt;&lt;li&gt;항목&lt;/li&gt;&lt;/ol&gt;</td>
   <td>&lt;i&gt;</td>
   <td>기울임꼴로 텍스트를 표시하거나 아이콘을 나타낼 때</td>
   <td>&lt;i&gt;기울임&lt;/i&gt;</td>
   <td>&lt;img&gt;</td>
   <td>이미지를 웹 페이지에 삽입할 때</td>
   <td>&lt;img src="image.jpg" alt="설명"&gt;</td>
 </tr>
 <tr>
   <td>&lt;table&gt;</td>
   <td>데이터를 행과 열로 구성된 표로 표시할 때</td>
   <td>&lt;table&gt;&lt;tr&gt;&lt;td&gt;셀&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;</td>
   <td>&lt;b&gt;</td>
   <td>텍스트를 굵게 표시할 때 (시각적 강조)</td>
   <td>&lt;b&gt;굵게&lt;/b&gt;</td>
   <td>&lt;iframe&gt;</td>
   <td>다른 HTML 페이지를 현재 페이지에 삽입할 때</td>
   <td>&lt;iframe src="url"&gt;&lt;/iframe&gt;</td>
 </tr>
 <td>&lt;style&gt;</td>
 <td>내부 CSS 스타일을 정의할 때 사용</td>
 <td>&lt;style&gt;body { color: black; }&lt;/style&gt;</td>
 <td>&lt;script&gt;</td>
 <td>JavaScript 코드를 포함하거나 외부 스크립트 파일을 연결할 때 사용</td>
 <td>&lt;script src="script.js"&gt;&lt;/script&gt;</td>
 <td>&lt;canvas&gt;</td>
 <td>그래픽을 그리기 위한 컨테이너로 사용</td>
 <td>&lt;canvas id="myCanvas"&gt;&lt;/canvas&gt;</td>
  </tr>
  <tr>
 <td>&lt;video&gt;</td>
 <td>비디오 콘텐츠를 삽입할 때 사용</td>
 <td>&lt;video src="video.mp4" controls&gt;&lt;/video&gt;</td>
 <td>&lt;mark&gt;</td>
 <td>텍스트를 하이라이트 할 때 사용</td>
 <td>&lt;mark&gt;하이라이트된 텍스트&lt;/mark&gt;</td>
 <td>&lt;progress&gt;</td>
 <td>작업의 진행 상태를 나타낼 때 사용</td>
 <td>&lt;progress value="70" max="100"&gt;&lt;/progress&gt;</td>
  </tr>
  <tr>
 <td>&lt;audio&gt;</td>
 <td>오디오 콘텐츠를 삽입할 때 사용</td>
 <td>&lt;audio src="audio.mp3" controls&gt;&lt;/audio&gt;</td>
 <td>&lt;kbd&gt;</td>
 <td>키보드 입력을 나타낼 때 사용</td>
 <td>Press &lt;kbd&gt;Ctrl&lt;/kbd&gt; + &lt;kbd&gt;C&lt;/kbd&gt; to copy</td>
 <td>&lt;meter&gt;</td>
 <td>알려진 범위 내의 스칼라 측정을 나타낼 때 사용</td>
 <td>&lt;meter value="0.6"&gt;60%&lt;/meter&gt;</td>
  </tr>
  <tr>
 <td>&lt;details&gt;</td>
 <td>사용자가 열고 닫을 수 있는 추가 세부 정보를 제공할 때 사용</td>
 <td>&lt;details&gt;&lt;summary&gt;제목&lt;/summary&gt;내용&lt;/details&gt;</td>
 <td>&lt;ruby&gt;</td>
 <td>동아시아 문자의 발음을 표시할 때 사용</td>
 <td>&lt;ruby&gt;漢&lt;rt&gt;かん&lt;/rt&gt;&lt;/ruby&gt;</td>
 <td>&lt;output&gt;</td>
 <td>계산이나 사용자 액션의 결과를 표시할 때 사용</td>
 <td>&lt;output name="result"&gt;계산 결과&lt;/output&gt;</td>
  </tr>
  <tr>
 <td>&lt;dialog&gt;</td>
 <td>대화 상자나 모달 윈도우를 생성할 때 사용</td>
 <td>&lt;dialog open&gt;대화 상자 내용&lt;/dialog&gt;</td>
 <td>&lt;datalist&gt;</td>
 <td>&lt;input&gt; 요소에 미리 정의된 옵션 목록을 제공할 때 사용</td>
 <td>&lt;input list="browsers"&gt;&lt;datalist id="browsers"&gt;&lt;option value="Chrome"&gt;&lt;/datalist&gt;</td>
 <td>&lt;picture&gt;</td>
 <td>반응형 이미지를 제공할 때 사용</td>
 <td>&lt;picture&gt;&lt;source srcset="img.webp"&gt;&lt;img src="img.jpg"&gt;&lt;/picture&gt;</td>
  </tr>
  <tr>
 <td>&lt;template&gt;</td>
 <td>페이지 로드 시 렌더링되지 않는 HTML 콘텐츠를 정의할 때 사용</td>
 <td>&lt;template&gt;&lt;p&gt;템플릿 내용&lt;/p&gt;&lt;/template&gt;</td>
 <td>&lt;link&gt;</td>
 <td>외부 리소스(주로 CSS 파일)를 연결할 때 사용</td>
 <td>&lt;link rel="stylesheet" href="styles.css"&gt;</td>
 <td>&lt;svg&gt;</td>
 <td>벡터 그래픽을 정의할 때 사용</td>
 <td>&lt;svg&gt;&lt;circle cx="50" cy="50" r="40"/&gt;&lt;/svg&gt;</td>
  </tr>
 <tr>
   <td>&lt;form&gt;</td>
   <td>사용자로부터 입력을 받아 제출하는 양식을 만들 때</td>
   <td>&lt;form&gt;폼 내용&lt;/form&gt;</td>
   <td>&lt;small&gt;</td>
   <td>작은 글씨로 부가 정보나 저작권 정보를 표시할 때</td>
   <td>&lt;small&gt;작은 글씨&lt;/small&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;section&gt;</td>
   <td>문서의 독립적인 섹션을 정의할 때</td>
   <td>&lt;section&gt;섹션 내용&lt;/section&gt;</td>
   <td>&lt;code&gt;</td>
   <td>컴퓨터 코드를 인라인으로 표시할 때</td>
   <td>&lt;code&gt;코드&lt;/code&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;article&gt;</td>
   <td>독립적으로 구분되거나 재사용 가능한 콘텐츠를 정의할 때</td>
   <td>&lt;article&gt;기사 내용&lt;/article&gt;</td>
   <td>&lt;cite&gt;</td>
   <td>저작물의 제목을 참조할 때</td>
   <td>&lt;cite&gt;출처&lt;/cite&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;header&gt;</td>
   <td>페이지나 섹션의 헤더 영역을 정의할 때</td>
   <td>&lt;header&gt;헤더 내용&lt;/header&gt;</td>
   <td>&lt;sub&gt;</td>
   <td>아래 첨자를 표시할 때 (예: 화학식)</td>
   <td>H&lt;sub&gt;2&lt;/sub&gt;O</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;footer&gt;</td>
   <td>페이지나 섹션의 푸터 영역을 정의할 때</td>
   <td>&lt;footer&gt;푸터 내용&lt;/footer&gt;</td>
   <td>&lt;sup&gt;</td>
   <td>위 첨자를 표시할 때 (예: 각주, 지수)</td>
   <td>x&lt;sup&gt;2&lt;/sup&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;nav&gt;</td>
   <td>주요 네비게이션 링크를 포함하는 영역을 정의할 때</td>
   <td>&lt;nav&gt;메뉴 항목&lt;/nav&gt;</td>
   <td>&lt;label&gt;</td>
   <td>폼 요소에 레이블을 붙일 때</td>
   <td>&lt;label for="id"&gt;레이블&lt;/label&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;aside&gt;</td>
   <td>페이지의 주요 내용과 간접적으로 연관된 내용을 표시할 때</td>
   <td>&lt;aside&gt;관련 내용&lt;/aside&gt;</td>
   <td>&lt;q&gt;</td>
   <td>짧은 인라인 인용구를 표시할 때</td>
   <td>&lt;q&gt;인용&lt;/q&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;main&gt;</td>
   <td>문서의 주요 콘텐츠를 정의할 때</td>
   <td>&lt;main&gt;주요 내용&lt;/main&gt;</td>
   <td>&lt;abbr&gt;</td>
   <td>약어나 두문자어를 표시할 때</td>
   <td>&lt;abbr title="설명"&gt;약어&lt;/abbr&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;figure&gt;</td>
   <td>이미지, 도표 등 독립적인 콘텐츠를 그룹화할 때</td>
   <td>&lt;figure&gt;&lt;img&gt;&lt;figcaption&gt;설명&lt;/figcaption&gt;&lt;/figure&gt;</td>
   <td>&lt;data&gt;</td>
   <td>내용과 기계가 읽을 수 있는 값을 연결할 때</td>
   <td>&lt;data value="값"&gt;표시 텍스트&lt;/data&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;blockquote&gt;</td>
   <td>다른 출처로부터의 긴 인용구를 표시할 때</td>
   <td>&lt;blockquote&gt;인용문&lt;/blockquote&gt;</td>
   <td>&lt;time&gt;</td>
   <td>날짜나 시간을 기계가 읽을 수 있는 형식으로 표시할 때</td>
   <td>&lt;time datetime="2023-05-15"&gt;5월 15일&lt;/time&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;pre&gt;</td>
   <td>형식이 미리 지정된 텍스트를 그대로 표시할 때</td>
   <td>&lt;pre&gt;  서식이    유지된   텍스트  &lt;/pre&gt;</td>
   <td>&lt;var&gt;</td>
   <td>수학 표현식이나 프로그래밍 문맥에서 변수를 나타낼 때</td>
   <td>&lt;var&gt;x&lt;/var&gt;</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;hr&gt;</td>
   <td>주제의 분리를 나타내는 수평선을 삽입할 때</td>
   <td>&lt;hr&gt;</td>
   <td>&lt;br&gt;</td>
   <td>텍스트 내에서 강제로 줄바꿈을 할 때</td>
   <td>텍스트&lt;br&gt;다음 줄</td>
   <td></td>
   <td></td>
   <td></td>
 </tr>
 <tr>
   <td>&lt;details&gt;</td>
   <td>사용자가 열고 닫을 수 있는 추가 세부 정보를 제공할 때</td>
   <td>&lt;details&gt;&lt;summary&gt;제목&lt;/summary&gt;내용&lt;/details&gt;</td>
   <td>&lt;wbr&gt;</td>
   <td>긴 단어나 URL에서 줄바꿈 위치를 제안할 때</td>
   <td>긴단어&lt;wbr&gt;계속</td>
   <td></td>
   <td></td>
   <td></td>
   <tr>
  
   <tr>
  <td>&lt;meta&gt;</td>
  <td>문서에 대한 메타데이터를 정의할 때 사용</td>
  <td>&lt;meta charset="UTF-8"&gt;</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
   </tr>
 </table>
  
</body>
</html>
