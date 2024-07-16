<!DOCTYPE html>
<html lang="en">

<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Document</title>
</head>
<style>

</style>
<!-- <a> 다른 페이지 이동할때 사용 -->
<!-- <button> 해당 페이지에서 동적인 무언가가 나타낼려할 때 -->

<body>
	<article>
		<h1>위니브 로그인</h1>
		<p>더편리해진 위니브에 오신 것을 환영 합니다.</p>
		<a href="">
			<img src="/image/weniv-logo.png" alt="weniv">로그인
		</a>
		<img src="/image/icon-lock.png" alt="">
		<a href="">아이디 | 비밀번호 찾기</a>
		<img src="/image/icon-user.png" alt="">
		<a href="">회원가입</a>
	</article>
	<!-- action 서버의 위치를 나타낸다  -->
	<!-- input 입력받은 데이터를 action 에 보내기위해  get 라는 method를 사용해야한다  -->
	<!-- input 도 submit 기능 있어 from 안에서 제출 버튼을 만들수 있다 -->
	<form method="get" action=" ">
		<!-- 사용자에게 입력 받은 정보를 제출할 때 -->
		<label for="user-age">나이</label>
		<input id="user-age" type="number" name="age">

	</form>
	<!-- enctype="multipart/form-data" 파일 보낼때 -->
	<!-- 양식
	 <form action="http://localhost:8000/" method="post" enctype="multipart/form-data">
		<input type="text" name="myTextField">
		<input type="checkbox" name="myCheckBox">Check</input>
		<input type="file" name="myFile">
		<button>Send the file</button>
	</form> -->
</body>

</html>